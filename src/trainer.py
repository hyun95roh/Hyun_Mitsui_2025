import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import pandas as pd
import logging 
from typing import Tuple, Optional, Dict, Any
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from transformers import Trainer as HFTrainer, TrainingArguments
from datasets import Dataset
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from accelerate import Accelerator
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
# From sibling modules:
from configs import TrainConfig 
from dataprep import WindowDataset, build_informer_datasets
from models import StandardScaler2D
from models import LSTMForecaster, CNNLSTMForecaster, TCNForecaster, FEDForecaster, InForecaster
from models import do_predict, arima_predict
from models import tune_weight_global, tune_weight_per_target, weighted_ensemble
from models import make_windows


class Trainer:
    """Base class for training deep learning models."""
    def __init__(self, cfg: TrainConfig, model: nn.Module = None, x_scaler: Optional[StandardScaler2D] = None):
        self.cfg = cfg
        # Validate device
        if hasattr(cfg, 'device'):
            if cfg.device in ["cuda", "cpu","gpu"]:
                self.device = cfg.device 
            else: 
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logging.warning(f"Invalid cfg.device={cfg.device}; set to {self.cfg.device}")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu" #self.cfg.device
        self.model = model.to(self.device) if model is not None else None
        self.x_scaler = x_scaler or StandardScaler2D()
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_state = None
        self.loss_tracker = {'total_epoch':cfg.epochs, 'epochs':[], 'train_loss':[], 'val_loss':[], 'best_loss':float('inf')}
        logging.info(f"Trainer initialized with device={self.device}")

    def make_dataloader(self, X: np.ndarray, Y: np.ndarray) -> DataLoader:
        """Create DataLoader from windowed data."""
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32)
        )
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)

    def build_scheduler(self, optimizer: torch.optim.Optimizer, steps_per_epoch: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler based on cfg.lr_policy."""
        if self.cfg.lr_policy == "clr":
            def lr_lambda(step):
                t = (step % steps_per_epoch) / max(1, steps_per_epoch - 1)
                return 0.1 + 0.9 * (1 - abs(2 * t - 1))  # Triangular CLR: 0.1x to 1.0x
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.cfg.lr_policy == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.cfg.lr, steps_per_epoch=steps_per_epoch,
                epochs=self.cfg.epochs, pct_start=0.3
            )
        elif self.cfg.lr_policy == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.cfg.scheduler_factor,
                patience=self.cfg.scheduler_patience, verbose=True
            )
        return None

    def masked_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Masked MSE loss for handling NaN targets."""
        mask = torch.isfinite(target)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        sq = (pred - target) ** 2 * mask
        denom = mask.sum().clamp_min(1)
        return sq.sum() / denom

    def train_one_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(self.device), yb.to(self.device).squeeze(1)
            pred = self.model(xb).squeeze(1)  # (B, num_targets)
            loss = self.masked_mse(pred, yb) if self.cfg.partially_finite_target else nn.MSELoss()(pred, yb)
            if torch.isnan(loss):
                for g in self.optimizer.param_groups:
                    g["lr"] *= 0.5
                continue
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
            self.optimizer.step()
            total_loss += loss.item() * len(xb)
            n_samples += len(xb)
        return total_loss / max(1, n_samples)

    def validate(self, val_loader: DataLoader) -> float:
        """Validate and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(self.device), yb.to(self.device).squeeze(1)
                pred = self.model(xb).squeeze(1)
                loss = nn.MSELoss()(pred, yb)
                total_loss += loss.item() * len(xb)
                n_samples += len(xb)
        return total_loss / max(1, n_samples)

    def train(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Train the model and return best validation loss."""
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(f"Expected X (N, T, F) and Y (N, output_len, num_targets), got shapes {X.shape}, {Y.shape}")
        if X.shape[1] != self.cfg.input_len or Y.shape[1] != self.cfg.output_len:
            raise ValueError(f"Expected X time dim {self.cfg.input_len}, Y time dim {self.cfg.output_len}, got {X.shape[1]}, {Y.shape[1]}")
        if X.shape[2] != self.cfg.input_size or Y.shape[2] != self.cfg.output_size:
            raise ValueError(f"Expected X feature dim {self.cfg.input_size}, Y target dim {self.cfg.output_size}, got {X.shape[2]}, {Y.shape[2]}")

        # Scale features
        X2d = X.reshape(-1, X.shape[-1])
        X2d = self.x_scaler.fit_transform(X2d)
        Xw = X2d.reshape(X.shape)
        Yw = Y

        # Split train/val
        N = Xw.shape[0]
        n_val = max(1, int(0.15 * N))
        Xtr, Ytr = Xw[:-n_val], Yw[:-n_val]
        Xva, Yva = Xw[-n_val:], Yw[-n_val:]

        train_loader = self.make_dataloader(Xtr, Ytr)
        val_loader = self.make_dataloader(Xva, Yva)

        # Optimizer
        opt_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "SGD": torch.optim.SGD}[self.cfg.optimizer]
        self.optimizer = opt_class(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Scheduler
        steps_per_epoch = max(1, len(train_loader))
        scheduler = self.build_scheduler(self.optimizer, steps_per_epoch)

        no_improve = 0
        for epoch in range(self.cfg.epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            # Logging 
            logging.info(f"Model: {self.cfg.model.model_type}, Epoch {epoch+1}/{self.cfg.epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
            self.loss_tracker['epochs'].append(epoch+1)
            self.loss_tracker['train_loss'].append(train_loss)
            self.loss_tracker['val_loss'].append(val_loss)
            self.loss_tracker['best_loss'] = self.best_val_loss 

            if scheduler:
                if self.cfg.lr_policy == "plateau":
                    scheduler.step(val_loss)
                elif self.cfg.lr_policy in ["clr", "onecycle"]:
                    scheduler.step()

            if no_improve >= self.cfg.patience:
                logging.info(f"Early stopping after {epoch+1} epochs")
                break

        return self.best_val_loss, self.loss_tracker

    def save_best_model(self, path: str) -> None:
        if self.best_state:
            torch.save(self.best_state, path)
            logging.info(f"Saved best model to {path}")

class LSTMTrainer(Trainer):
    """Trainer for LSTMForecaster."""
    def __init__(self, cfg: TrainConfig):
        model = LSTMForecaster(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            output_len=cfg.output_len,
            num_targets=cfg.output_size
        )
        super().__init__(cfg, model)

class CNNLSTMTrainer(Trainer):
    """Trainer for CNNLSTMForecaster."""
    def __init__(self, cfg: TrainConfig):
        model = CNNLSTMForecaster(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            output_len=cfg.output_len,
            num_targets=cfg.output_size,
            channels=cfg.channels,
            kernel_size=cfg.kernel_size,
            num_conv_layers=cfg.num_conv_layers
        )
        super().__init__(cfg, model)

class TCNTrainer(Trainer):
    """Trainer for TCNForecaster."""
    def __init__(self, cfg: TrainConfig):
        model = TCNForecaster(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            output_len=cfg.output_len,
            num_targets=cfg.output_size,
            channels=cfg.channels,
            kernel_size=cfg.tcn_kernel,
            blocks=cfg.blocks
        )
        super().__init__(cfg, model)

class FEDTrainer(Trainer):
    """Trainer for FEDForecaster."""
    def __init__(self, cfg: TrainConfig):
        model = FEDForecaster(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            output_len=cfg.output_len,
            num_targets=cfg.output_size,
            decom_kernel_size=cfg.decom_kernel_size,
            top_k_modes=cfg.top_k_modes
        )
        super().__init__(cfg, model)

class TSTPTrainer(Trainer):
    """Trainer for TimeSeriesTransformerForPrediction."""
    def __init__(self, cfg: TrainConfig):
        config = TimeSeriesTransformerConfig(
            prediction_length=cfg.output_len,
            context_length=cfg.input_len,
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            num_dynamic_real_features=cfg.input_size,
            num_time_features=3,
            num_static_categorical_features=0,
            num_static_real_features=0,
            scaling=None,  # Disable scaling to avoid loc/scale issues
            lags_sequence=[1]  # Minimal lags to simplify input processing
        )
        logging.info(f"TimeSeriesTransformerConfig: {config}")
        model = TimeSeriesTransformerForPrediction(config)
        super().__init__(cfg, model)

    def train(self, X: np.ndarray, Y: np.ndarray) -> float:
        # Initialize Accelerator
        accelerator = Accelerator(device_placement=True)
        logging.info(f"Accelerator device: {accelerator.device}")
        self.model = accelerator.prepare(self.model)

        # Scale features
        X2d = X.reshape(-1, X.shape[-1])
        X2d = self.x_scaler.fit_transform(X2d)
        Xw = X2d.reshape(X.shape)
        Yw = Y

        # Split train/val
        N = Xw.shape[0]
        n_val = max(1, int(0.15 * N))
        Xtr, Ytr = Xw[:-n_val], Yw[:-n_val]
        Xva, Yva = Xw[-n_val:], Yw[-n_val:]
        logging.info(f"TSTP train shapes: Xtr={Xtr.shape}, Ytr={Ytr.shape}, Val shapes: Xva={Xva.shape}, Yva={Yva.shape}")

        # Generate time features
        def generate_time_features(seq_len, num_features=3):
            time_indices = np.arange(seq_len)
            time_features = np.zeros((seq_len, num_features), dtype=np.float32)
            time_features[:, 0] = time_indices / seq_len
            time_features[:, 1] = np.sin(2 * np.pi * time_indices / seq_len)
            time_features[:, 2] = np.cos(2 * np.pi * time_indices / seq_len)
            return time_features

        # Generate time features and observed masks
        time_features_tr = generate_time_features(self.cfg.input_len, num_features=3)
        time_features_va = generate_time_features(self.cfg.input_len, num_features=3)
        mask_tr = np.isfinite(Xtr).astype(np.float32)
        mask_va = np.isfinite(Xva).astype(np.float32)

        # Convert to torch.Tensor and validate
        Xtr_tensor = torch.tensor(Xtr, dtype=torch.float32)
        Ytr_tensor = torch.tensor(Ytr, dtype=torch.float32)
        Xva_tensor = torch.tensor(Xva, dtype=torch.float32)
        Yva_tensor = torch.tensor(Yva, dtype=torch.float32)
        time_features_tr_tensor = torch.tensor(np.repeat(time_features_tr[None, :, :], Xtr.shape[0], axis=0), dtype=torch.float32)
        time_features_va_tensor = torch.tensor(np.repeat(time_features_va[None, :, :], Xva.shape[0], axis=0), dtype=torch.float32)
        mask_tr_tensor = torch.tensor(mask_tr, dtype=torch.float32)
        mask_va_tensor = torch.tensor(mask_va, dtype=torch.float32)

        # Validate tensors
        for name, tensor in [
            ('Xtr_tensor', Xtr_tensor),
            ('Ytr_tensor', Ytr_tensor),
            ('Xva_tensor', Xva_tensor),
            ('Yva_tensor', Yva_tensor),
            ('time_features_tr_tensor', time_features_tr_tensor),
            ('time_features_va_tensor', time_features_va_tensor),
            ('mask_tr_tensor', mask_tr_tensor),
            ('mask_va_tensor', mask_va_tensor)
        ]:
            if tensor is None:
                logging.error(f"{name} is None")
                raise ValueError(f"{name} is None")
            logging.info(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}")

        # Try HFTrainer first to test if custom loop is the issue
        train_data = Dataset.from_dict({
            'past_values': Xtr_tensor.numpy(),
            'future_values': Ytr_tensor.numpy(),
            'past_time_features': time_features_tr_tensor.numpy(),
            'past_observed_mask': mask_tr_tensor.numpy()
        })
        val_data = Dataset.from_dict({
            'past_values': Xva_tensor.numpy(),
            'future_values': Yva_tensor.numpy(),
            'past_time_features': time_features_va_tensor.numpy(),
            'past_observed_mask': mask_va_tensor.numpy()
        })

        logging.info(f"Train dataset keys: {train_data.column_names}")
        logging.info(f"Val dataset keys: {val_data.column_names}")

        # HFTrainer setup
        try:
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=self.cfg.epochs,
                per_device_train_batch_size=self.cfg.batch_size,
                eval_strategy='epoch',
                save_strategy='epoch',
                disable_tqdm=False,
                report_to='none'
            )
            hf_trainer = HFTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_data,
                eval_dataset=val_data
            )
            hf_trainer.train()
            eval_results = hf_trainer.evaluate()
            val_loss = eval_results['eval_loss']
            self.best_val_loss = val_loss
            self.best_state = self.model.state_dict()
            logging.info(f"TSTP validation loss (HFTrainer): {val_loss:.6f}")
            return val_loss
        except Exception as e:
            logging.warning(f"HFTrainer failed: {str(e)}. Falling back to custom training loop.")

        # Custom training loop as fallback
        def collate_fn(batch):
            past_values = torch.stack([item[0] for item in batch])
            future_values = torch.stack([item[1] for item in batch])
            past_time_features = torch.stack([item[2] for item in batch])
            past_observed_mask = torch.stack([item[3] for item in batch])
            return past_values, future_values, past_time_features, past_observed_mask

        train_dataset = TensorDataset(Xtr_tensor, Ytr_tensor, time_features_tr_tensor, mask_tr_tensor)
        val_dataset = TensorDataset(Xva_tensor, Yva_tensor, time_features_va_tensor, mask_va_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        sample_batch = next(iter(train_loader))
        logging.info(f"Sample train batch shapes: past_values={sample_batch[0].shape}, "
                     f"future_values={sample_batch[1].shape}, "
                     f"past_time_features={sample_batch[2].shape}, "
                     f"past_observed_mask={sample_batch[3].shape}")
        logging.info(f"Sample train batch values: past_values[0,0,:5]={sample_batch[0][0,0,:5]}, "
                     f"past_time_features[0,:2,:]={sample_batch[2][0,:2,:]}, "
                     f"past_observed_mask[0,0,:5]={sample_batch[3][0,0,:5]}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        for epoch in range(self.cfg.epochs):
            self.model.train()
            total_loss = 0.0
            n_samples = 0
            for batch in train_loader:
                past_values, future_values, past_time_features, past_observed_mask = [
                    b.to(self.device) for b in batch
                ]
                for name, tensor in [
                    ('past_values', past_values),
                    ('future_values', future_values),
                    ('past_time_features', past_time_features),
                    ('past_observed_mask', past_observed_mask)
                ]:
                    if tensor is None:
                        logging.error(f"Batch {name} is None")
                        raise ValueError(f"Batch {name} is None")
                    logging.info(f"Batch {name} shape: {tensor.shape}, first few values: {tensor[0,0,:5]}")

                outputs = self.model(
                    past_values=past_values,
                    past_time_features=past_time_features,
                    past_observed_mask=past_observed_mask,
                    future_values=future_values
                )
                loss = outputs.loss
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                total_loss += loss.item() * len(past_values)
                n_samples += len(past_values)
            train_loss = total_loss / max(1, n_samples)

            self.model.eval()
            val_loss = 0.0
            n_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    past_values, future_values, past_time_features, past_observed_mask = [
                        b.to(self.device) for b in batch
                    ]
                    outputs = self.model(
                        past_values=past_values,
                        past_time_features=past_time_features,
                        past_observed_mask=past_observed_mask,
                        future_values=future_values
                    )
                    loss = outputs.loss
                    val_loss += loss.item() * len(past_values)
                    n_samples += len(past_values)
            val_loss = val_loss / max(1, n_samples)

            logging.info(f"Model: {self.cfg.model.model_type}, Epoch {epoch+1}/{self.cfg.epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = self.model.state_dict()

        self.model.load_state_dict(self.best_state)
        logging.info(f"TSTP validation loss: {self.best_val_loss:.6f}")
        return self.best_val_loss


# =======================================================================
# TimesFM additions (trainer)
# -----------------------------------------------------------------------
# A thin wrapper that uses your existing finetuning framework to finetune
# TimesFM on sliding-window datasets built in dataprep.py.
# =======================================================================
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader, Subset 
from timesfm.pytorch_patched_decoder import create_quantiles  # used by finetuner

# Local imports from your repo
from models import load_timesfm_model
from dataprep import build_timesfm_loaders_from_series, TimesFMSlidingWindowDataset, TimesFMDataSpec
from finetuning.finetuning_torch import TimesFMFinetuner, FinetuningConfig

@dataclass
class TimesFMTrainerConfig:
    model_name: str = "google/timesfm-1.0-200m-pytorch"
    context_length: int = 512
    horizon_length: int = 5     # 128 was original setting for google/timesfm-1.0-200m-pytorch
    freq_type: int = 0          # 0=daily (example), align with your mapping
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 0.0
    use_quantile_loss: bool = False
    # logging / ckpt
    out_dir: str = "checkpoints_timesfm"
    single_step_pred: bool = True  # True -> compare step 0 (or choose lag in your loop)
    head_hidden: int = 64
    head_dropout: float = 0.0

class TimesFMTrainer:
    """High-level trainer to finetune a TimesFM backbone per target (or multi-task if desired)."""
    def __init__(self, cfg: TimesFMTrainerConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(cfg.out_dir, exist_ok=True)
        # Load TimesFM backbone
        self.model = load_timesfm_model(
            model_name=self.cfg.model_name,
            device=self.device,
            horizon_len=self.cfg.horizon_length,
            context_len=self.cfg.context_length,
            batch_size=self.cfg.batch_size,
            local_ckpt='../checkpoints/timesfm-1.0-200m/torch_model.ckpt'
        )
        logging.info(f"Loaded TimesFM model '{cfg.model_name}' on {self.device}")

    def _make_finetuner(self) -> TimesFMFinetuner:
        fcfg = FinetuningConfig(
            batch_size=self.cfg.batch_size,
            num_epochs=self.cfg.epochs,
            learning_rate=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            use_quantile_loss=self.cfg.use_quantile_loss,
            quantiles=None,
            device=self.device,
            distributed=False,
            gpu_ids=[0] if torch.cuda.is_available() else [],
            master_port=29500,
            master_addr="127.0.0.1",
            use_wandb=False,
            wandb_project=None,
            log_every_n_steps=50,
            val_check_interval=1.0,
        )
        return TimesFMFinetuner(model=self.model, config=fcfg, rank=0, loss_fn=None, logger=logging.getLogger(__name__))

    def train_one_series(self, series, target_name: str = "target"):
        series = torch.as_tensor(series, dtype=torch.float32)
        if series.ndim == 1:
            series = series[:, None]

        ds_all = TimesFMSlidingWindowDataset(
            series, 
            context_length=self.cfg.context_length, 
            horizon_length=self.cfg.horizon_length, 
            freq_type=self.cfg.freq_type
            )
        n = len(ds_all)
        n_val = max(1, int(n * 0.15))
        n_train = max(1, n - n_val)
        train_ds = Subset(ds_all, list(range(0, n_train)))
        val_ds   = Subset(ds_all, list(range(n - n_val, n)))

        finetuner = self._make_finetuner()
        result = finetuner.finetune(
            train_dataset=train_ds,
            val_dataset=val_ds,
            single_step_pred=self.cfg.single_step_pred,
        )

        # you can still choose where to save here:
        tgt_dir = os.path.join(self.cfg.out_dir, f"finetuned_{target_name}")
        os.makedirs(tgt_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(tgt_dir, "finetuned.pt"))
        return result

    def train_many(self, series_list: List[torch.Tensor] | List[list], target_names: Optional[List[str]] = None):
        if target_names is None:
            target_names = [f"target_{i}" for i in range(len(series_list))]
        assert len(target_names) == len(series_list)
        histories = []
        for name, series in zip(target_names, series_list):
            logging.info(f"Fine-tuning TimesFM for {name} ...")
            result = self.train_one_series(series, target_name=name)
            histories.append({"name": name, **result})        
        try:
            self.consolidate_heads(num_targets=len(series_list))
        except Exception as e:
            logging.warning(f"Head consolidation skipped: {e}")
        return histories

    def consolidate_heads(self, num_targets: int = 424, bank_filename: str = "timesfm_heads_bank.pt"):
        """
        Pack all finetuned per-target heads into one state_dict file.
        Looks for: {out_dir}/finetuned_target_{j}/head.pt
        Saves to : {out_dir}/{bank_filename}
        """

        H = self.cfg.horizon_length
        bank = nn.ModuleDict({f"t{j}": make_timesfm_head(H, self.cfg.head_hidden, self.cfg.head_dropout)
                            for j in range(num_targets)})

        for j in range(num_targets):
            head_path = os.path.join(self.cfg.out_dir, f"finetuned_target_{j}", "head.pt")
            state = torch.load(head_path, map_location="cpu")
            bank[f"t{j}"].load_state_dict(state)

        save_path = os.path.join(self.cfg.out_dir, bank_filename)
        torch.save(bank.state_dict(), save_path)
        return save_path


############################
# Informer model 
############################
class InTrainer(Trainer):
    """Trainer for InForecaster."""
    def __init__(self, cfg: TrainConfig, criterion=nn.MSELoss()):
        def _cfg_get(key, default=None):
            # 1) Try attribute (dataclass TrainConfig)
            if hasattr(cfg, key):
                return getattr(cfg, key)
            # 2) Try Hydra DictConfig layout: cfg.params.key
            try:
                return cfg.model.params.get(key, default)
            except Exception:
                pass
            # 3) Fallback: flat dict-like
            try:
                return cfg.get(key, default)
            except Exception:
                return default
        model_params = {
            'input_size':   _cfg_get('input_size', 1672),
            'output_size':  _cfg_get('output_size', 424),   # unifies c_out -> output_size
            'input_len':    _cfg_get('input_len', 64),      # Informer: seq_len
            'label_len':    _cfg_get('label_len', 32),
            'output_len':   _cfg_get('output_len', 5),      # Informer: out_len
            'hidden_size':  _cfg_get('hidden_size', 256),   # Informer: d_model
            'nhead':        _cfg_get('nhead', 8),           # Informer: n_heads
            'num_layers':   _cfg_get('num_layers', 3),      # Informer: e_layers
            'd_layers':     _cfg_get('d_layers', 2),
            'd_ff':         _cfg_get('d_ff', 512),          # <-- FIX: use d_ff (not dim_feedforward)
            'dropout':      _cfg_get('dropout', 0.1),
            'attn':         _cfg_get('attn', 'prob'),
            'distil':       _cfg_get('distil', True),
        }
        self.criterion = criterion 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"InTrainer initialized with model_kwargs: {model_params}")
        mp = model_params
        forecaster_kwargs = dict(
            input_size = mp['input_size'],
            enc_in     = mp['input_size'],
            dec_in     = mp['output_size'],
            c_out      = mp['output_size'],
            seq_len    = mp['input_len'],
            label_len  = mp['label_len'],
            out_len    = mp['output_len'],
            d_model    = mp['hidden_size'],
            n_heads    = mp['nhead'],
            e_layers   = mp['num_layers'],
            d_layers   = mp['d_layers'],
            d_ff       = mp['d_ff'],
            dropout    = mp['dropout'],
            attn       = mp['attn'],
            distil     = mp['distil'],
        )
        self.model = InForecaster(**forecaster_kwargs)
        super().__init__(cfg, model=self.model)

    def _unpack_batch(self, batch):
        # ADDED: unified unpack for 5-tuple (with time feats) or 3-tuple (no time feats)
        if len(batch) == 5:
            x_enc, x_dec, x_mark_enc, x_mark_dec, y = batch
        elif len(batch) == 3:
            x_enc, x_dec, y = batch
            x_mark_enc = x_mark_dec = None
        else:
            raise ValueError(f"Unexpected batch tuple length: {len(batch)}")
        return x_enc, x_mark_enc, x_dec, x_mark_dec, y


    def train(self, X: np.ndarray = None, Y: np.ndarray = None, train_loader: DataLoader = None, val_loader: DataLoader = None) -> float:
        if train_loader is None or val_loader is None:
            if X is None or Y is None:
                raise ValueError("Must provide either (train_loader, val_loader) or (X, Y)")
            if X.ndim != 2 or Y.ndim != 2:
                raise ValueError(f"Expected raw X (N, F) and Y (N, T), got shapes {X.shape}, {Y.shape}")
            train_loader, val_loader, _ = build_informer_datasets(
                self.cfg, X, Y, use_time_feats=True
            )

        # Optimizer
        opt_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW, "SGD": torch.optim.SGD}[self.cfg.optimizer]
        self.optimizer = opt_class(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Scheduler
        steps_per_epoch = max(1, len(train_loader))
        scheduler = self.build_scheduler(self.optimizer, steps_per_epoch)

        no_improve = 0
        for epoch in range(self.cfg.epochs):
            self.model.train()
            total_loss = 0.0
            n_samples = 0
            for batch in train_loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y = self._unpack_batch(batch)

                x_enc = x_enc.to(self.device, non_blocking=True)
                x_dec = x_dec.to(self.device, non_blocking=True)
                y     = y.to(self.device, non_blocking=True)
                if x_mark_enc is not None:
                    x_mark_enc = x_mark_enc.to(self.device, non_blocking=True)
                    x_mark_dec = x_mark_dec.to(self.device, non_blocking=True)

                pred = self.model(x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
                if pred.size(0) != y.size(0) and pred.size(0) == y.size(0) * self.cfg.output_len:
                    # Recover from accidental batch*time flattening
                    pred = pred.view(y.size(0), self.cfg.output_len, -1)
                # Expect multi-horizon: pred (B,T,C), y (B,T,C)
                if pred.dim() == 2: pred = pred.unsqueeze(1)     # (B,1,C) fallback
                if y.dim() == 2:
                    raise RuntimeError(f"Labels must be (B,T,C); got {y.shape}. "
                                    f"Ensure InformerWindowDataset returns y with shape (T,C).")

                B, T, C = pred.shape
                assert y.shape == (B, T, C), f"pred {pred.shape} vs y {y.shape}"

                loss = self.criterion(pred.reshape(B*T, C), y.reshape(B*T, C))
                #loss = self.masked_mse(pred, y) if self.cfg.partially_finite_target else nn.MSELoss()(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                self.optimizer.step()
                total_loss += loss.item() * len(x_enc)
                n_samples += len(x_enc)
            train_loss = total_loss / max(1, n_samples)

            val_loss = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.cfg.patience:
                logging.info(f"Early stopping at epoch {epoch}")
                break

            if scheduler and self.cfg.lr_policy == "plateau":
                scheduler.step(val_loss)
            elif scheduler:
                scheduler.step()

            logging.info(f"Model: {self.cfg.model.model_type}, Epoch {epoch+1}/{self.cfg.epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        return self.best_val_loss

    def validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total, denom = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x_enc, x_mark_enc, x_dec, x_mark_dec, y = self._unpack_batch(batch)
                x_enc = x_enc.to(self.device, non_blocking=True)
                x_dec = x_dec.to(self.device, non_blocking=True)
                y     = y.to(self.device, non_blocking=True)
                if x_mark_enc is not None:
                    x_mark_enc = x_mark_enc.to(self.device, non_blocking=True)
                    x_mark_dec = x_mark_dec.to(self.device, non_blocking=True)

                pred = self.model(x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
                if pred.dim() == 2: pred = pred.unsqueeze(1)

                B, T, C = pred.shape
                assert y.shape == (B, T, C), f"pred {pred.shape} vs y {y.shape}"
                loss = self.criterion(pred.reshape(B*T, C), y.reshape(B*T, C))
                total += loss.item() * (B*T)      # weight by batch horizons
                denom += (B*T)
        return total / max(1, denom)




from models import DeRiTSBackbone
from utils import mae_loss
from spectral import freq_derivative_multiplier, apply_freq_derivative, inverse_freq_derivative
from spectral import ComplexLinear, ComplexDepthwiseConvFreq, BandMask

class DERTrainer(Trainer):
    """
    Minimal trainer specialized for DERITS:
    - accepts TensorDataset or custom dataset
    - supports AMP, grad clipping
    - adds optional spectral regularizers
    """
    def __init__(self, cfg, train_ds, val_ds, device="cuda"):
        super().__init__(cfg) 
        import torch
        import torch.nn as nn
        self.cfg = cfg
        if hasattr(cfg,'device'):
            self.device = cfg.device 
        else:
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=cfg.num_workers, drop_last=False)
        self.val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)

        class ComplexDropout(nn.Module):
            def __init__(self, p: float = 0.1, inplace: bool = False):
                super().__init__()
                self.p = p
                self.inplace = inplace
            def forward(self, zc: torch.Tensor) -> torch.Tensor:
                # zc: complex tensor
                zr = F.dropout(zc.real, p=self.p, training=self.training, inplace=self.inplace)
                zi = F.dropout(zc.imag, p=self.p, training=self.training, inplace=self.inplace)
                return torch.complex(zr, zi)


        # ---- tiny DERITS backbone here (import your version if you put it elsewhere)
        class FourierConvBlock(nn.Module):
            def __init__(self, channels, kernel_size=5, dropout=0.1):
                super().__init__()
                self.cmix = ComplexLinear(channels, channels)
                self.dw = ComplexDepthwiseConvFreq(channels, kernel_size=kernel_size)
                #self.dropout = nn.Dropout(dropout)
                self.cdrop = ComplexDropout(dropout) 
            def forward(self, zc):  # complex [B,C,F]
                res = zc
                zc = self.cmix(zc) 
                zc = self.dw(zc)
                return res + self.cdrop(zc) 

        class _DerivativeBranch(nn.Module):
            def __init__(self, order, L, C_in, D, depth=2, kernel_size=5, dropout=0.1):
                super().__init__()
                self.order, self.L, self.C_in, self.D = order, L, C_in, D
                self.proj_in = ComplexLinear(C_in, C_in)
                self.blocks = nn.ModuleList([FourierConvBlock(C_in, kernel_size, dropout) for _ in range(depth)])
                self.band = None
                self.head = ComplexLinear(C_in, D)

            def _ensure_band(self, F, device):
                if self.band is None:
                    self.band = BandMask(self.C_in, F).to(device)
                    # push the first ~1% lowest-frequency bins down at init to avoid DC dominance
                    k0 = max(1, int(0.01 * F))
                    with torch.no_grad():
                        self.band.mask_logits[..., :k0] = -8.0  # sigmoid(-8) ~ 0.0003
            
            def forward(self, x):  # x: [B,L,C_in]
                # --- keep complex path in full precision to avoid ComplexHalf issues ---
                with torch.amp.autocast("cuda", enabled=False):
                    xt = x.transpose(1, 2)                         # [B,C,L] float32

                    if not torch.isfinite(xt).all():
                        logging.warning("Non-finite values in input x; replacing with zeros")
                        xt = torch.nan_to_num(xt, nan=0.0, posinf=1e6, neginf=-1e6)        
                    
                    X  = torch.fft.rfft(xt, dim=-1)                # [B,C,F] complex64
                    Fbins = X.shape[-1]
                    self._ensure_band(Fbins, X.device)

                    Xd = apply_freq_derivative(X, order=self.order, L=self.L)  # complex64
                    Xd = self.proj_in(Xd)                                      # complex64
                    Xd, _ = self.band(Xd)                                      # complex64
                    for blk in self.blocks:
                        Xd = blk(Xd)                                           # complex64
                    Yd = self.head(Xd)                                         # [B,D,F] complex64
                    Yc = inverse_freq_derivative(Yd, order=self.order, L=self.L)
                    yt = torch.fft.irfft(Yc, n=self.L, dim=-1)                 # [B,D,L] float32
                    yt = torch.clamp(yt, min=-10.0, max=10.0)
                    y  = yt.transpose(1, 2).contiguous()                       # [B,L,D] float32

                    # last-resort guard (prevents rare NaNs from infecting the decoder)
                    y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

                return y


        class DeRiTSBackbone(nn.Module):
            def __init__(self, L, H, C_in, D, orders=(0,1,2), depth=2, kernel_size=5, dropout=0.1, rnn_hidden_mul=2):
                super().__init__()
                self.model_type = 'DERITS'
                self.num_targets = D
                self.L, self.H, self.C_in, self.D = L, H, C_in, D
                orders = [int(o) for o in orders]  # ← coerce here
                self.branches = nn.ModuleList([_DerivativeBranch(o, L, C_in, D, depth, kernel_size, dropout)
                               for o in orders])
                self.fusion = nn.Parameter(torch.ones(len(orders)) / max(1, len(orders)))
                self.temporal_head = nn.GRU(input_size=D, hidden_size=rnn_hidden_mul*D, batch_first=True)
                self.proj_out = nn.Linear(rnn_hidden_mul*D, D)
            def forward(self, x):  # x: [B,L,C_in]
                ys = [b(x) for b in self.branches]        # [B,L,D] each
                W = torch.softmax(self.fusion, dim=0)
                Y = sum(w * y for w, y in zip(W, ys))     # [B,L,D]
                seq, _ = self.temporal_head(Y)            # [B,L,r]
                h = seq[:, -1:, :]
                preds = []
                for _ in range(self.H):
                    step = self.proj_out(h.squeeze(1))    # [B,D]
                    preds.append(step.unsqueeze(1))
                    h, _ = self.temporal_head(step.unsqueeze(1))
                return torch.cat(preds, dim=1)            # [B,H,D]

        self.model = DeRiTSBackbone(
            L=cfg.L, H=cfg.H, C_in=cfg.C_in, D=cfg.D,
            orders=tuple(cfg.orders), depth=cfg.depth, kernel_size=cfg.kernel_size,
            dropout=cfg.dropout, rnn_hidden_mul=getattr(cfg, "rnn_hidden_mul", 2)
        ).to(self.device)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        self.crit = nn.L1Loss()
        self.cuda = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = torch.amp.GradScaler(self.cuda, enabled=cfg.amp)

    # ---- optional spectral regularizers
    def _bandmask_l1(self):
        reg = 0.0
        for m in self.model.modules():
            if hasattr(m, "band") and m.band is not None and hasattr(m.band, "mask_logits"):
                reg = reg + m.band.mask_logits.abs().mean()
        return getattr(self.cfg, "band_l1", 0.0) * reg

    def _highfreq_penalty(self):
        w = getattr(self.cfg, "hf_weight", 0.0)
        if w <= 0: return 0.0
        frac = getattr(self.cfg, "hf_fraction", 0.2)
        reg = 0.0
        for m in self.model.modules():
            if hasattr(m, "band") and m.band is not None:
                mask = m.band.mask_logits.sigmoid()   # [1,C,F]
                F = mask.shape[-1]; k0 = int((1.0 - frac) * F)
                if k0 < F: reg = reg + mask[..., k0:].mean()
        return w * reg

    def _unpack(self, batch):
        # Accept either dict or tuple batches
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            x, y = batch
        return x.to(self.device), y.to(self.device)

    def train_one_epoch(self, train=True):
        loader = self.train_loader if train else self.val_loader
        self.model.train(mode=train)
        total, n = 0.0, 0
        for batch in loader:
            x, y = self._unpack(batch)          # x: [B,L,C], y: [B,H,D]
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                logging.warning(f"Non-finite values in batch: x={torch.isfinite(x).all().item()}, y={torch.isfinite(y).all().item()}")
            with torch.amp.autocast(self.cuda, enabled=self.cfg.amp):
                yhat = self.model(x)
                yhat = torch.clamp(yhat, min=-1.0, max=1.0)
                loss = self.crit(yhat, y) + self._bandmask_l1() + self._highfreq_penalty()
                if not torch.isfinite(loss):
                    logging.warning("Non-finite loss encountered; skipping batch.")
                    loss = self.masked_mse(yhat, y) + self._bandmask_l1() + self._highfreq_penalty()
                    if not torch.isfinite(loss):
                        logging.warning("Masked MSE still non-finite; skipping batch")
                        continue
            
            if train:
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if getattr(self.cfg, "grad_clip", 0.0) > 0:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.scaler.step(self.opt); self.scaler.update()
            total += loss.item() * len(x); n += len(x)
        return total / max(1, n)

