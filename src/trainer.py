import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import logging 
from typing import Tuple, Optional, Dict, Any

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from configs import TrainConfig 
from dataprep import WindowDataset
from models import StandardScaler2D
from models import LSTMForecaster, CNNLSTMForecaster, TCNForecaster, FEDForecaster
from models import do_predict, arima_predict
from models import tune_weight_global, tune_weight_per_target, weighted_ensemble
from models import make_windows
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction


class Trainer:
    """Base class for training deep learning models."""
    def __init__(self, cfg: TrainConfig, model: nn.Module, x_scaler: Optional[StandardScaler2D] = None):
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.x_scaler = x_scaler or StandardScaler2D()
        self.device = cfg.device
        self.history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_state = None

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
        """
        Train the model and return best validation loss.
        """
        # Validate input shapes
        if X.ndim != 3 or Y.ndim != 3:
            raise ValueError(f"Expected X (N, T, F) and Y (N, output_len, num_targets), got shapes {X.shape}, {Y.shape}")
        if X.shape[1] != self.cfg.input_len or Y.shape[1] != self.cfg.output_len:
            raise ValueError(f"Expected X time dim {self.cfg.input_len}, Y time dim {self.cfg.output_len}, got {X.shape[1]}, {Y.shape[1]}")
        if X.shape[2] != self.cfg.input_size or Y.shape[2] != self.cfg.output_size:
            raise ValueError(f"Expected X feature dim {self.cfg.input_size}, Y target dim {self.cfg.output_size}, got {X.shape[2]}, {Y.shape[2]}")
        
        logging.info(f"Trainer input shapes: X={X.shape}, Y={Y.shape}")
        print(f"{self.model.model_type}")

        # Scale features
        X2d = X.reshape(-1, X.shape[-1])  # (N*T, F)
        X2d = self.x_scaler.fit_transform(X2d)
        X2d = np.nan_to_num(X2d, nan=0.0, posinf=0.0, neginf=0.0)
        Xw = X2d.reshape(X.shape)  # (N, T, F)
        Yw = Y  # Already windowed (N, output_size, num_targets)
        
        # Split train/val
        N = Xw.shape[0]
        n_val = max(1, int(0.15 * N))
        Xtr, Ytr = Xw[:-n_val], Yw[:-n_val]
        Xva, Yva = Xw[-n_val:], Yw[-n_val:]
        logging.info(f"Train shapes: Xtr={Xtr.shape}, Ytr={Ytr.shape}, Val shapes: Xva={Xva.shape}, Yva={Yva.shape}")

        # Create dataloaders
        train_loader = self.make_dataloader(Xtr, Ytr)
        val_loader = self.make_dataloader(Xva, Yva)

        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = self.build_scheduler(self.optimizer, len(train_loader))

        # Training loop
        bad_epochs = 0
        for epoch in range(self.cfg.epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            print(f"Epoch {epoch+1}/{self.cfg.epochs} | train {train_loss:.6f} | val {val_loss:.6f} | lr {self.optimizer.param_groups[0]['lr']:.2e}")

            if val_loss + 1e-8 < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.patience:
                    print(f"Early stopping: {bad_epochs} bad epochs")
                    break

            if scheduler and self.cfg.lr_policy == "plateau":
                scheduler.step(val_loss)
            elif scheduler:
                scheduler.step()

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        self.model.eval()
        return self.best_val_loss

    def save_best_model(self, path: str):
        """Save the best model state to path."""
        if self.best_state:
            torch.save(self.best_state, path)
            print(f"Saved best model to {path}")

    def get_best_val_loss(self) -> float:
        """Return the best validation loss."""
        return self.best_val_loss

class LSTMTrainer(Trainer):
    """Trainer for LSTMForecaster."""
    def __init__(self, cfg: TrainConfig):
        model = LSTMForecaster(
            input_size=cfg.input_size, 
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            output_len=cfg.output_len,
            num_targets=cfg.output_size # num_targets = output_size
        )
        super().__init__(cfg, model)

class CNNLSTMTrainer(Trainer):
    """Trainer for CNNLSTMForecaster."""
    def __init__(self, cfg: TrainConfig):
        model = CNNLSTMForecaster(
            input_size=cfg.input_size,   # input_size = timesteps of features
            num_targets=cfg.output_size, # num_targets = output_size
            conv_channels=cfg.channels,
            kernel_size=cfg.kernel_size,
            num_conv_layers=cfg.num_conv_layers,
            lstm_hidden=cfg.hidden_size,
            lstm_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        super().__init__(cfg, model)


class TCNTrainer(Trainer):
    """
    Trainer for TCNForecaster, inheriting from base Trainer class.
    Initializes TCNForecaster with parameters from TrainConfig and handles training.
    
    Args:
        cfg: TrainConfig with hyperparameters (e.g., hidden_size, num_layers).
        input_size: Number of input features (F).
        num_targets: Number of target variables.
    """
    def __init__(self, cfg: TrainConfig):
        logging.info(f"TCNTrainer init: input_size={cfg.input_size}, num_targets={cfg.output_size}")
        channels = [cfg.channels] * cfg.blocks
        model = TCNForecaster(
            input_size=cfg.input_size,
            num_targets=cfg.output_size, # num_targets = output_size
            channels_list=channels,
            kernel_size=getattr(cfg, "tcn_kernel", 3),
            dropout=cfg.dropout
        )
        super().__init__(cfg, model)


class FEDTrainer(Trainer):
    """
    Trainer for FEDForecaster, inheriting from base Trainer class.
    Initializes FEDForecaster with parameters from TrainConfig and handles training.
    
    Args:
        cfg: TrainConfig with hyperparameters (e.g., hidden_size, num_layers).
        input_size: Number of input features (F).
        num_targets: Number of target variables.
    """
    def __init__(self, cfg: TrainConfig):
        logging.info(f"FEDTrainer init: input_size={cfg.input_size}, num_targets={cfg.output_size}")
        model = FEDForecaster(
            input_size=cfg.input_size,  # Use provided input_size (e.g., 1672)
            d_model=cfg.hidden_size,
            nhead=getattr(cfg, "nhead", 8),
            num_layers=cfg.num_layers,
            dim_feedforward=getattr(cfg, "dim_feedforward", cfg.hidden_size * 4),
            dropout=cfg.dropout,
            num_targets=cfg.output_size, # num_targets = output_size 
            decom_kernel_size=getattr(cfg, "decom_kernel_size", 25),
            top_k_modes=getattr(cfg, "top_k_modes", 5)
        )
        super().__init__(cfg, model)

class TSTPTrainer(Trainer):
    def __init__(self, cfg: TrainConfig): 
        logging.info(f"TransformerTrainer init: input_size={cfg.input_size}, num_targets={cfg.output_size}")
        config = TimeSeriesTransformerConfig(
            input_size = cfg.input_size, 
            prediction_length = cfg.output_len, 
            context_length = cfg.input_len, 
            d_model = cfg.hidden_size, 
            num_attention_heads = cfg.nhead, 
            num_layers = cfg.num_layers, 
            dim_feedforward = cfg.dim_feedforward, 
            dropout = cfg.dropout 
        )
        model = TimeSeriesTransformerForPrediction(config)
        super().__init__(cfg, model)




#################
# Ensemble Trainer 
#################
def run_ensemble(
    targets_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    cfg: TrainConfig,
    feature_cols=None,
    use_multivariate: bool=False,
    arima_jobs: int=1,
    weight_mode: str="global",
    tune_window: int=200,
    max_lag: int=4
) -> Dict[str, np.ndarray]:
    """Run ensemble of LSTM and ARIMA models."""
    if use_multivariate:
        feature_cols = feature_cols or [c for c in prices_df.columns if c != "date_id"]
        X = prices_df.set_index("date_id").loc[targets_df.index, feature_cols].to_numpy(float)
        Y = targets_df.to_numpy(float)
        
        trainer = LSTMTrainer(cfg, input_size=X.shape[1], num_targets=Y.shape[1])
        val_loss = trainer.train(X, Y)
        model, x_scaler = trainer.model, trainer.x_scaler

        # Tuning window
        tune_end = len(Y) - (max_lag + 1)
        if tune_end <= cfg.input_len + 5:
            raise ValueError("Not enough non-NaN history to tune.")
        tune_start = max(0, tune_end - tune_window)
        idx_tune = slice(tune_start, tune_end)
        X_tune, Y_tune = X[idx_tune], Y[idx_tune]

        # Rolling predictions
        yl_list, ya_list, yt_list = [], [], []
        for t in range(cfg.input_len, len(Y_tune)):
            print(f" - Loop: {t}/{len(Y_tune)-1}")
            X_hist = X_tune[:t]
            Y_hist = Y_tune[:t]
            yhat_l = do_predict(model, x_scaler, Y_hist=Y_hist, input_len=cfg.input_len, X_hist=X_hist)
            yhat_a = arima_predict(pd.DataFrame(Y_hist, columns=targets_df.columns), horizon=1, n_jobs=arima_jobs)
            yt = Y_tune[t]
            yl_list.append(yhat_l)
            ya_list.append(yhat_a)
            yt_list.append(yt)

        YL = np.vstack(yl_list)
        YA = np.vstack(ya_list)
        YT = np.vstack(yt_list)
        finite_rows = np.isfinite(YT).all(axis=1) & np.isfinite(YL).all(axis=1) & np.isfinite(YA).all(axis=1)
        YTf, YLf, YAf = YT[finite_rows], YL[finite_rows], YA[finite_rows]

        w = 1.0 if not finite_rows.any() else (tune_weight_per_target(YTf, YLf, YAf) if weight_mode == "per_target" else tune_weight_global(YTf, YLf, YAf))

        yhat_next_l = do_predict(model, x_scaler, Y_hist=Y, input_len=cfg.input_len, X_hist=X)
        yhat_next_a = arima_predict(targets_df, horizon=1, n_jobs=arima_jobs)
        yhat_next = weighted_ensemble(yhat_next_l, yhat_next_a, w)
        
        return dict(pred_lstm=yhat_next_l, pred_arima=yhat_next_a, pred_ens=yhat_next, w=w)

    else:
        preds_l, preds_a, preds_e, ws = [], [], [], []
        for col in targets_df.columns:
            y = targets_df[col].to_numpy(float)
            trainer = LSTMTrainer(cfg, input_size=1, num_targets=1)
            val_loss = trainer.train(y[:, None], y[:, None])
            model, x_scaler = trainer.model, trainer.x_scaler
            yhat_l = do_predict(model, x_scaler, Y_hist=y, input_len=cfg.input_len)
            yhat_a = arima_predict(targets_df[[col]], horizon=1, n_jobs=1)[0]
            w = 0.5
            preds_l.append(yhat_l[0])
            preds_a.append(yhat_a)
            ws.append(w)
            preds_e.append(weighted_ensemble(yhat_l[0], yhat_a, w))
        
        return dict(
            pred_lstm=np.array(preds_l, dtype=float),
            pred_arima=np.array(preds_a, dtype=float),
            pred_ens=np.array(preds_e, dtype=float),
            w=np.array(ws, dtype=float)
        )