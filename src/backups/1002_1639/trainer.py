import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging 
from typing import Tuple, Optional, Dict, Any

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
from dataprep import WindowDataset
from models import StandardScaler2D
from models import LSTMForecaster, CNNLSTMForecaster, TCNForecaster, FEDForecaster
from models import do_predict, arima_predict
from models import tune_weight_global, tune_weight_per_target, weighted_ensemble
from models import make_windows


class Trainer:
    """Base class for training deep learning models."""
    def __init__(self, cfg: TrainConfig, model: nn.Module = None, x_scaler: Optional[StandardScaler2D] = None):
        self.cfg = cfg
        # Validate device
        if cfg.device is None or cfg.device not in ["cuda", "cpu"]:
            self.cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.warning(f"Invalid cfg.device={cfg.device}; set to {self.cfg.device}")
        self.device = self.cfg.device
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
                self.best_state = self.model.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            # Logging 
            logging.info(f"Model: {self.cfg.model_type}, Epoch {epoch+1}/{self.cfg.epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
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
            d_model=cfg.hidden_size,
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
            d_model=cfg.hidden_size,
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

            logging.info(f"Model: {self.cfg.model_type}, Epoch {epoch+1}/{self.cfg.epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = self.model.state_dict()

        self.model.load_state_dict(self.best_state)
        logging.info(f"TSTP validation loss: {self.best_val_loss:.6f}")
        return self.best_val_loss


# timesfm_trainer.py
from typing import Optional
from configs import SimpleCfg
from finetuning_torch import FinetuningConfig, TimesFMFinetuner
from timesfm.pytorch import TimesFm


class TimesFMTrainer:
    """Loads google/timesfm-1.0-200m-pytorch and finetunes via your TimesFMFinetuner."""
    def __init__(self, cfg: SimpleCfg):
        self.cfg = cfg
        self.model = TimesFm.from_pretrained("google/timesfm-1.0-200m-pytorch").to(cfg.device)

        self.fcfg = FinetuningConfig(
            batch_size=cfg.batch_size,
            num_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay,
            freq_type=cfg.freq_type,
            use_quantile_loss=cfg.use_quantile_loss,
            quantiles=cfg.quantiles,
            device=cfg.device,
            distributed=False,
            gpu_ids=[0],
            master_port="12358",
            master_addr="localhost",
            use_wandb=False,
            log_every_n_steps=50,
            val_check_interval=0.5,
        )
        self.finetuner = TimesFMFinetuner(self.model, self.fcfg)

    def _make_loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=shuffle,
                          drop_last=False, num_workers=self.cfg.num_workers, pin_memory=True)

    def fit(self, train_ds, val_ds=None, single_step_pred: Optional[bool]=None) -> dict:
        if single_step_pred is None:
            single_step_pred = self.cfg.single_step_pred
        return self.finetuner.finetune(train_dataset=train_ds, val_dataset=val_ds,
                                       single_step_pred=single_step_pred)

