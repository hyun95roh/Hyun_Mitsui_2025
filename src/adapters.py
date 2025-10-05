import pandas as pd 
import numpy as np 
import warnings 
from dataclasses import dataclass 
from typing import Tuple, List, Dict, Callable, Optional, Protocol, Iterable  
import logging, traceback 
import torch 

# from sibiling modules
from models import do_predict
from models import TCNForecaster
from models import load_timesfm_heads_bank 

######################
# Adapters 
######################
class NextStepAdapter(Protocol):
    """
    Any model you add (LSTM, CNN+LSTM, Prophet, FEDformer, XGBoost, ...) should implement this.
    """
    def predict_next(
        self,
        X_hist: pd.DataFrame,             # dense features, index=date_id, columns in trained order
        Y_hist: Optional[pd.DataFrame],   # optional targets history aligned to X_hist (if the model uses it)
        input_len: int
    ) -> np.ndarray:                      # returns (num_targets,)
        ...



class LSTMAdapter:
    def __init__(self, model, x_scaler=None, target_cols: List[str]=None):
        self.model, self.x_scaler, self.target_cols = model, x_scaler, target_cols
        logging.info(f"LSTMAdapter initialized with target_cols: {target_cols}")

    def predict_next(self, X_hist: pd.DataFrame, Y_hist: Optional[pd.DataFrame], input_len: int, lag: int = 1) -> tuple[np.ndarray, dict]:
        """
        Predict next step with date_id tracking.
        
        Args:
            X_hist: DataFrame with date_id column.
            Y_hist: Optional DataFrame (ignored for multivariate).
            input_len: Number of historical steps.
            lag: Forecast horizon.
        
        Returns:
            Tuple of (prediction: np.ndarray, metadata: dict with date_id, target_date_id).
        """
        if len(self.target_cols) != self.model.num_targets:
            logging.error(f"Target columns mismatch: expected {self.model.num_targets}, got {len(self.target_cols)}: {self.target_cols}")
            raise ValueError(f"Target columns mismatch in {self.model.model_type}Adapter")        
        try:
            X_full = X_hist.to_numpy(dtype=float)
            date_id = X_hist['date_id'].iloc[-1] if 'date_id' in X_hist.columns else None
            Y_dummy = np.empty((len(X_full), 1), dtype=np.float32)
            logging.info(f"LSTMAdapter predict_next: target_cols={self.target_cols}, X_hist shape={X_full.shape}, date_id={date_id}, lag={lag}")
            return do_predict(self.model, self.x_scaler, Y_dummy, input_len=input_len, X_hist=X_full, date_id=date_id, lag=lag)
        except Exception as e:
            logging.error(f"LSTMAdapter predict_next failed: {str(e)}")
            raise

class CNNLSTMAdapter:
    def __init__(self, model, x_scaler, target_cols):
        self.model = model.eval()
        self.x_scaler = x_scaler
        self.target_cols = target_cols
        logging.info(f"CNNLSTMAdapter initialized with target_cols: {target_cols}")

    def predict_next(self, X_hist, Y_hist, input_len: int, lag: int = 1) -> tuple[np.ndarray, dict]:
        if len(self.target_cols) != self.model.num_targets:
            logging.error(f"Target columns mismatch: expected {self.model.num_targets}, got {len(self.target_cols)}: {self.target_cols}")
            raise ValueError(f"Target columns mismatch in {self.model.model_type}Adapter")        
        try:
            X_full = X_hist.to_numpy(dtype=float)
            date_id = X_hist['date_id'].iloc[-1] if 'date_id' in X_hist.columns else None
            Y_dummy = np.empty((len(X_full), 1), dtype=np.float32)
            logging.info(f"CNNLSTMAdapter predict_next: target_cols={self.target_cols}, X_hist shape={X_full.shape}, date_id={date_id}, lag={lag}")
            pred, metadata = do_predict(self.model, self.x_scaler, Y_dummy, input_len=input_len, X_hist=X_full, date_id=date_id, lag=lag)
            # Ensure metadata uses the correct lag
            metadata['lag'] = lag 
            metadata['target_date_id'] = date_id + lag + 1 if date_id is not None else None
            return pred, metadata
        except Exception as e:
            logging.error(f"CNNLSTMAdapter predict_next failed: {str(e)}")
            raise

class _IdentityScaler2D:
    """Fallback if scaler.pkl is missing; behaves like a StandardScaler on 2D (N,F) but no-ops."""
    def fit(self, X2d): return self
    def transform(self, X2d): return X2d
    def fit_transform(self, X2d): return X2d

class TCNAdapter:
    """
    Loads a trained TCNForecaster checkpoint and emits a full 424-dim prediction vector
    given the latest input_len steps of X_hist.

    Expects checkpoint dir to contain:
      - lstm.pt (or tcn.pt)           # the model state_dict (your save_ckp used 'lstm.pt'; name doesn't matter)
      - scaler.pkl                     # joblib StandardScaler2D (2D)
      - feature_cols.json              # list of training-time feature column names
      - target_cols.json               # list of target column names in training order
      - (optional) model_params.json     # dict with model parameters (e.g., channels/kernel for TCN)
    """
    def __init__(self, model, x_scaler, target_cols, ckpt_dir: str = None, device: str = None):
        self.ckpt_dir = ckpt_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Use load_ckp to load all necessary components
        if ckpt_dir:
            self.model, self.x_scaler, self.feature_cols, self.target_cols, self.model_params = load_ckp(
                dir=ckpt_dir,
                model_class=TCNForecaster,
                model_kwargs={}, # model_kwargs will be built inside load_ckp based on model_params.json
                device=self.device
            )
        else: 
            self.model = model
            self.x_scaler = x_scaler 
            self.target_cols = target_cols
            self.feature_cols = None   # None in non-ckpt case (will handle in predict_next)
            self.model_params = {}     # Dafault empty to avoid AttributeError


        # Ensure the model is in evaluation mode
        self.model.eval()

        # Use loaded model_params to set adapter attributes if needed (e.g., input_len)
        self.input_len = self.model_params.get("input_len", 64)
        logging.info(f"TCNAdapter initialized with target_cols: {self.target_cols}")

    def predict_next(self, X_hist: pd.DataFrame, Y_hist: pd.DataFrame | np.ndarray, input_len: int, lag: int = 1) -> tuple[np.ndarray, dict]:
        if len(self.target_cols) != self.model.num_targets:
            logging.error(f"Target columns mismatch: expected {self.model.num_targets}, got {len(self.target_cols)}: {self.target_cols}")
            raise ValueError(f"Target columns mismatch in {self.model.model_type}Adapter")
        try:
            # Select features
            if self.feature_cols is not None:
                if not all(c in X_hist.columns for c in self.feature_cols):
                    missing = [c for c in self.feature_cols if c not in X_hist.columns]
                    raise KeyError(f"X_hist is missing {len(missing)} expected features. e.g. {missing[:5]}")
                cols_to_use = self.feature_cols
            else:
                cols_to_use = [c for c in X_hist.columns if c != 'date_id']
                expected_features = self.model.input_size
                if len(cols_to_use) != expected_features:
                    logging.warning(f"Feature count mismatch: expected {expected_features}, got {len(cols_to_use)}. Adjusting...")
                    if len(cols_to_use) < expected_features:
                        pad_cols = [f"pad_{i}" for i in range(expected_features - len(cols_to_use))]
                        X_hist = X_hist.copy()
                        for col in pad_cols:
                            X_hist[col] = 0.0
                        cols_to_use += pad_cols
                    else:
                        cols_to_use = cols_to_use[:expected_features]

            current_input_len = input_len
            date_id = X_hist['date_id'].iloc[-1] if 'date_id' in X_hist.columns else None

            # Prepare input block
            X_block = X_hist[cols_to_use].iloc[-current_input_len:].to_numpy(dtype=float)
            logging.info(f"TCNAdapter predict_next: target_cols={self.target_cols}, X_block shape={X_block.shape}, date_id={date_id}, lag={lag}")
            if X_block.shape[0] < current_input_len:
                pad = current_input_len - X_block.shape[0]
                if X_block.shape[0] == 0:
                    return np.zeros(len(self.target_cols), dtype=float), {"date_id": date_id, "target_date_id": None, "lag": lag}
                first = X_block[0:1, :]
                X_block = np.vstack([first.repeat(pad, axis=0), X_block])

            # Call do_predict
            return do_predict(
                model=self.model,
                X_scaler=self.x_scaler,
                Y_hist=Y_hist,
                input_len=input_len,
                X_hist=X_block,
                date_id=date_id,
                lag=lag
            )
        except Exception as e:
            logging.error(f"TCNAdapter predict_next failed: {str(e)}\n{traceback.format_exc()}")
            raise


# Add to adapters.py after existing adapters
class FEDAdapter:
    """
    Adapter for FEDForecaster, implementing NextStepAdapter protocol.
    Handles one-step-ahead predictions using the last input_len rows of X_hist.
    
    Args:
        model: Trained FEDForecaster instance.
        x_scaler: StandardScaler2D or None for input scaling.
        target_cols: List of target column names in training order.
    """
    def __init__(self, model, x_scaler=None, target_cols: List[str]=None):
        self.model = model.eval()
        self.x_scaler = x_scaler if x_scaler is not None else _IdentityScaler2D()
        self.target_cols = target_cols
        logging.info(f"FEDAdapter initialized with target_cols: {target_cols}")

    def predict_next(self, 
                     X_hist: pd.DataFrame, 
                     Y_hist: Optional[pd.DataFrame], 
                     input_len: int, 
                     lag: int = 1) -> np.ndarray:
        """
        Predict next step using FEDForecaster.
        
        Args:
            X_hist: DataFrame with features (index=date_id, columns in trained order).
            Y_hist: Optional DataFrame of target history (ignored for multivariate).
            input_len: Number of historical steps to use.
        
        Returns:
            np.ndarray of shape (num_targets,) with one-step-ahead predictions.
        """
        if len(self.target_cols) != self.model.num_targets:
            logging.error(f"Target columns mismatch: expected {self.model.num_targets}, got {len(self.target_cols)}: {self.target_cols}")
            raise ValueError(f"Target columns mismatch in {self.model.model_type}Adapter")
        try:
            X_full = X_hist.to_numpy(dtype=float)
            date_id = X_hist['date_id'].iloc[-1] if 'date_id' in X_hist.columns else None
            Y_dummy = np.empty((len(X_full), 1), dtype=np.float32)
            logging.info(f"FEDAdapter predict_next: target_cols={self.target_cols}, X_hist shape={X_full.shape}, date_id={date_id}, lag={lag}")
            return do_predict(self.model, self.x_scaler, Y_dummy, input_len=input_len, X_hist=X_full, date_id=date_id, lag=lag)
        except Exception as e:
            logging.error(f"FEDAdapter predict_next failed: {str(e)}")
            raise

class TSTPAdapter:
    def __init__(self, model, x_scaler=None, target_cols: List[str]=None):
        self.model = model.eval()
        self.x_scaler = x_scaler if x_scaler is not None else _IdentityScaler2D()
        self.target_cols = target_cols
        logging.info(f"TransformerAdapter initialized with target_cols: {target_cols}")

    def predict_next(self, X_hist: pd.DataFrame, Y_hist: Optional[pd.DataFrame], input_len: int, lag: int = 1) -> tuple[np.ndarray, dict]:
        if len(self.target_cols) != self.model.config.prediction_length:
            logging.error(f"Target columns mismatch: expected {self.model.config.prediction_length}, got {len(self.target_cols)}")
            raise ValueError(f"Target columns mismatch in TransformerAdapter")
        try:
            X_full = X_hist.to_numpy(dtype=float)
            date_id = X_hist['date_id'].iloc[-1] if 'date_id' in X_hist.columns else None
            X_block = X_full[-input_len:]
            if self.x_scaler:
                X_block = self.x_scaler.transform(X_block)
            X_block = np.nan_to_num(X_block, nan=0.0, posinf=0.0, neginf=0.0)
            x = torch.tensor(X_block, dtype=torch.float32, device=self.model.device).unsqueeze(0)
            with torch.no_grad():
                output = self.model.generate(input_ids=x, prediction_length=1)
                pred = output.sequences.cpu().numpy()[0, -1, :]
            target_date_id = date_id + lag + 1 if date_id is not None else None
            metadata = {"date_id": date_id, "target_date_id": target_date_id, "lag": lag}
            return pred, metadata
        except Exception as e:
            logging.error(f"TransformerAdapter predict_next failed: {str(e)}\n{traceback.format_exc()}")
            raise


class TimesFMAdapter:
    """
    Adapter for TimesFM model, implementing NextStepAdapter protocol.
    Handles one-step-ahead predictions for all 424 targets by processing each target series individually.
    
    Args:
        model: Trained or fine-tuned TimesFM model (nn.Module from load_timesfm_model).
        x_scaler: StandardScaler2D or None for input scaling.
        target_cols: List of target column names in training order.
        context_length: Number of historical steps for TimesFM context (default 512).
        horizon_length: Number of forecast steps (default 1 for one-step-ahead).
        freq_type: Integer code for frequency (default 0 for daily).
    """
    def __init__(self, model, x_scaler=None, target_cols: List[str]=None, 
                 context_length: int=512, horizon_length: int=1, freq_type: int=0,
                 heads_bank_path: str | None =None, head_hidden: int = 64, head_dropout: float = 0.0
                 ):
        self.model = model.eval()
        self.x_scaler = x_scaler if x_scaler is not None else _IdentityScaler2D()
        self.target_cols = target_cols
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.freq_type = freq_type
        self.device = next(model.parameters()).device
        # Load packed per-target heads (t0..t423) if provided
        self.heads = None
        if heads_bank_path:
            self.heads = load_timesfm_heads_bank(
                bank_path=heads_bank_path,
                num_targets=424,
                horizon_len=self.horizon_length,
                hidden=head_hidden,
                dropout=head_dropout,
                device=str(self.device),
            )
    
        logging.info(f"TimesFMAdapter initialized with target_cols: {len(target_cols) if target_cols else 0}, context_length: {context_length}, device: {self.device}")

    def predict_next(self, X_hist: pd.DataFrame, Y_hist: Optional[pd.DataFrame], input_len: int, lag: int = 1) -> tuple[np.ndarray, dict]:
        """
        Predict next step using TimesFM for each target series.
        
        Args:
            X_hist: DataFrame with features (index=date_id, columns in trained order).
            Y_hist: Optional DataFrame of target history (used as input series for TimesFM).
            input_len: Number of historical steps to use (must be >= context_length).
            lag: Forecast horizon (currently supports lag=1 for one-step-ahead).
        
        Returns:
            Tuple of (prediction: np.ndarray of shape (num_targets,), metadata: dict).
        """
        if len(self.target_cols) != 424:  # Hard-coded for Mitsui challenge
            logging.error(f"Target columns mismatch: expected 424, got {len(self.target_cols)}: {self.target_cols}")
            raise ValueError("Target columns mismatch in TimesFMAdapter")
        
        try:
            date_id = X_hist['date_id'].iloc[-1] if 'date_id' in X_hist.columns else None
            target_date_id = date_id + lag + 1 if date_id is not None else None
            metadata = {"date_id": date_id, "target_date_id": target_date_id, "lag": lag}

            # Ensure input_len matches context_length
            if input_len < self.context_length:
                logging.warning(f"input_len {input_len} is less than context_length {self.context_length}; adjusting to {self.context_length}")
                input_len = self.context_length

            # Prepare input: Use Y_hist if provided, else fall back to X_hist
            if Y_hist is not None and not Y_hist.empty:
                series_df = Y_hist[self.target_cols]
            else:
                # Fall back to X_hist (assume target_cols are subset of X_hist columns)
                missing_cols = [c for c in self.target_cols if c not in X_hist.columns]
                if missing_cols:
                    logging.warning(f"Missing {len(missing_cols)} target columns in X_hist; using zeros for {missing_cols[:5]}...")
                    series_df = pd.DataFrame(0.0, index=X_hist.index, columns=self.target_cols)
                    for col in self.target_cols:
                        if col in X_hist.columns:
                            series_df[col] = X_hist[col]
                else:
                    series_df = X_hist[self.target_cols]

            # Ensure sufficient history
            if len(series_df) < input_len:
                pad_rows = input_len - len(series_df)
                pad_df = pd.DataFrame(0.0, index=range(-pad_rows, 0), columns=series_df.columns)
                series_df = pd.concat([pad_df, series_df], axis=0)

            # -------- batched contexts for all targets --------
            B = len(self.target_cols)
            ctx_list = []
            for col in self.target_cols:
                arr = series_df[col].values[-self.context_length:].astype(np.float32)  # last context window
                arr = arr.reshape(self.context_length, 1)
                arr = self.x_scaler.transform(arr)  # keep shape (L,1)
                ctx_list.append(torch.from_numpy(arr))
            x_context = torch.stack(ctx_list, dim=0).to(self.device)        # [B, L, 1]
            x_padding = torch.ones(B, self.context_length, dtype=torch.float32, device=self.device)
            freq = torch.full((B,), int(self.freq_type), dtype=torch.long, device=self.device)

            # -------- single TimesFM forward --------
            with torch.no_grad():
                out = self.model(x_context, x_padding, freq)                 # [B, P, H, 1+Q]
                mean = out[..., 0][:, 0, :]                                  # [B, H], first patch mean

            # -------- apply per-target heads (bank) --------
            if self.heads is None:
                # No calibration heads → use base mean
                adjusted = mean
            else:
                rows = []
                for j in range(B):
                    rows.append(self.heads[f"t{j}"](mean[j]))               # [H]
                adjusted = torch.stack(rows, dim=0)                          # [B, H]

            # pick the step for this lag (0..4)
            if lag < 0 or lag >= self.horizon_length:
                raise ValueError(f"lag={lag} out of range for horizon_length={self.horizon_length}")
            pred_vec = adjusted[:, lag]                                       # [B]
            predictions = pred_vec.detach().cpu().float().numpy()
            
            ####
            pred_array = np.asarray(predictions, dtype=np.float32)
            if not np.isfinite(pred_array).all():
                logging.warning("Non-finite predictions detected; replacing with zeros")
                pred_array = np.nan_to_num(pred_array, nan=0.0, posinf=0.0, neginf=0.0)

            logging.info(f"TimesFMAdapter predict_next: predicted for date_id {date_id}, shape={pred_array.shape}")
            return pred_array, metadata

        except Exception as e:
            logging.error(f"TimesFMAdapter predict_next failed: {str(e)}\n{traceback.format_exc()}")
            return np.zeros(424, dtype=float), {"date_id": date_id, "target_date_id": None, "lag": lag}

