# models.py
import pandas as pd 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils import weight_norm 
from math import sqrt 
import torch 
import warnings 
import pmdarima as pm 
import joblib
import json 
import os 
import logging, traceback 


from collections import defaultdict 
from dataclasses import dataclass 
from typing import Tuple, List, Dict, Callable, Optional, Protocol, Iterable  
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from time import time 


######################
# Save checkpoints
######################
def _to_list(cols: Optional) -> Optional[list]:
    if cols is None:
        return None
    # works for list/tuple/Index
    return list(cols)

def save_ckp(dir: str,
             model,                     # torch nn.Module
             scaler,                    # e.g., StandardScaler2D or None (joblib-picklable)
             X_cols: Optional=None,
             Y_cols: Optional=None) -> None:
    """
    Saves:
      dir/lstm.pt           (model state_dict)
      dir/scaler.pkl        (if scaler is not None)
      dir/feature_cols.json (training feature order)
      dir/target_cols.json  (training target order)
    """
    os.makedirs(dir, exist_ok=True)

    # model
    torch.save(model.state_dict(), os.path.join(dir, "lstm.pt"))

    # scaler
    if scaler is not None:
        joblib.dump(scaler, os.path.join(dir, "scaler.pkl"))

    # columns (don’t use truthiness on pd.Index; just check not None)
    x_list = _to_list(X_cols)
    y_list = _to_list(Y_cols)

    if x_list is not None:
        with open(os.path.join(dir, "feature_cols.json"), "w") as f:
            json.dump(list(map(str, x_list)), f)

    if y_list is not None:
        with open(os.path.join(dir, "target_cols.json"), "w") as f:
            json.dump(list(map(str, y_list)), f)

def load_ckp(dir: str, model_class, model_kwargs, device: str = 'cpu'):
    """
    Loads model state, scaler, and columns.
    Returns: (model, scaler, feat_cols, target_cols, model_params)
    """
    # rebuild empty model and load weights
    model = model_class(**model_kwargs)
    state = torch.load(os.path.join(dir, "lstm.pt"), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # scaler (optional)
    scaler_path = os.path.join(dir, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    # columns
    with open(os.path.join(dir, "feature_cols.json")) as f:
        feature_cols = json.load(f)
    with open(os.path.join(dir, "target_cols.json")) as f:
        target_cols = json.load(f)
    
    # model_params (optional)
    model_params_path = os.path.join(dir, "model_params.json")
    model_params = json.load(open(model_params_path)) if os.path.exists(model_params_path) else {}

    return model, scaler, feature_cols, target_cols, model_params

######################
# Scalers 
######################
class MinMaxScalerTS:
    """Fit on train only. Transform/Inverse-transform 1D arrays."""
    def __init__(self, clip=True, eps=1e-9):
        self.min_, self.max_, self.clip, self.eps = None, None, clip, eps
    def fit(self, x: np.ndarray):
        x = np.asarray(x).reshape(-1)
        self.min_, self.max_ = np.nanmin(x), np.nanmax(x)
        return self
    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.min_) / (self.max_ - self.min_ + self.eps)
        return np.clip(z, 0.0, 1.0) if self.clip else z
    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return z * (self.max_ - self.min_ + self.eps) + self.min_


class StandardScaler2D:
    """Columnwise (features) scaler for 2D arrays."""
    def __init__(self, eps: float = 1e-8):
        self.mu = None
        self.sd = None
        self.eps = eps

    def fit(self, X: np.ndarray):
        self.mu = np.nanmean(X, axis=0)
        self.sd = np.nanstd(X, axis=0)
        self.sd = np.where(self.sd < self.eps, 1.0, self.sd)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.sd + self.mu

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


####################
# Dataset
####################


######################
# LSTM model 
######################
class LSTMForecaster(nn.Module):
    """
    Unified: supports input_size>=1, num_targets>=1, output_len>=1
    - If univariate: input_size=1, num_targets=1
    - If multivariate: input_size=F, num_targets=T
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, output_len: int, num_targets: int):
        super().__init__()
        self.model_type = 'LSTM'
        self.output_len, self.num_targets = output_len, num_targets #num_targets = output_size 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_size, num_targets * output_len)

    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, H)
        last = out[:, -1, :]            # (B, H)
        y = self.head(last)             # (B, TGT*out_len)
        return y.view(-1, self.output_len, self.num_targets)  # (B, out_len, TGT)



def do_predict(
    model: nn.Module,
    X_scaler: Optional[StandardScaler2D],
    Y_hist: Optional[np.ndarray],
    input_len: int,
    X_hist: Optional[np.ndarray] = None,
    date_id: Optional[int] = None,
    lag: int = 1
) -> tuple[np.ndarray, dict]:
    """
    One-step ahead prediction using the last input_len rows.
    If X_hist is None -> univariate AR (use Y_hist as the single input feature).
    
    Args:
        model: Trained model (e.g., LSTMForecaster, FEDForecaster).
        X_scaler: Scaler for input features.
        Y_hist: Historical targets (N,) or (N, TGT).
        input_len: Number of input time steps.
        X_hist: Historical features (N, F) or None.
        date_id: Last date_id of X_hist (for tracking prediction timing).
        lag: Forecast horizon (e.g., lag=1 predicts 2 steps ahead).
    
    Returns:
        Tuple of (prediction: np.ndarray (num_targets,), metadata: dict with date_id info).
    """
    #print(" --- Model:", model.model_type)
    t_s = time()
    Y_hist = np.asarray(Y_hist, dtype=np.float32)
    if Y_hist.ndim == 1:
        Y_hist = Y_hist[:, None]  # (N, 1)
    logging.info(f"do_predict: Y_hist shape={Y_hist.shape}, input_len={input_len}, lag={lag}")    

    if X_hist is None:
        X_hist = Y_hist  # univariate AR

    X_hist = np.asarray(X_hist, dtype=np.float32)
    if len(X_hist) < input_len:
        logging.warning(f"Insufficient history length: {len(X_hist)} < {input_len}")
        return np.full((model.num_targets,), np.nan, dtype=float), {"date_id": date_id, "target_date_id": None}

    Xw = X_hist[-input_len:]  # (T, F)
    if not np.isfinite(Xw).all():
        logging.warning("Non-finite values in input history")
        return np.full((model.num_targets,), np.nan, dtype=float), {"date_id": date_id, "target_date_id": None}

    if X_scaler is not None:
        Xw = X_scaler.transform(Xw)

    device = next(model.parameters()).device
    logging.info(f"do_predict: Model on device {device}, Input shape=(1, {input_len}, {Xw.shape[1]}), date_id={date_id}")
    
    try:
        x = torch.tensor(Xw, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, F)
        with torch.no_grad():
            out = model(x).cpu().numpy()  # (1, out_len, num_targets)
    except Exception as e:
        logging.error(f"do_predict failed: {str(e)}\n{traceback.format_exc()}")
        raise 

    # Compute target date_id
    target_date_id = date_id + lag + 1 if date_id is not None else None
    logging.info(f"Prediction for target_date_id={target_date_id} (lag={lag}, input date_id={date_id})")
    
    metadata = {"date_id": date_id, "target_date_id": target_date_id, "lag": lag}
    #print(f"Prediction --- done! | Duration: {time()-t_s}")
    return out[0, lag, :], metadata  # (num_targets,), metadata #pick step = lag (0~4)

# ARIMA forcaster 
@dataclass 
class ARIMAConfig:
    seasonal: bool = False 
    m: int = 1 # season length (12=monthly, 5=weekday, etc)
    info_criteron: str = 'aicc' # aic, bic, aicc 
    stepwise: bool = True 
    max_p: int = 5 
    max_q: int = 5 
    max_d: int = 2 
    max_P: int = 2 
    max_Q: int = 2 
    max_D: int = 1 

class ArimaForecaster:
    def __init__(self, cfg: ARIMAConfig):
        self.cfg = cfg 
        self.model = None 
        self.model_type = 'ARIMA'

    def fit(self, train_vals: np.ndarray):
        train_vals = pd.Series(train_vals).dropna().values
        if len(train_vals) < 20:
            self.model = ("naive", 0.0)  # fallback
            return self
        self.model = pm.auto_arima(
            train_vals, 
            seasonal= self.cfg.seasonal, 
            m = self.cfg.m,
            information_criterion = self.cfg.info_criteron,
            stepwise = self.cfg.stepwise, 
            max_p = self.cfg.max_p, 
            max_q = self.cfg.max_q, 
            max_d = self.cfg.max_d, 
            max_P = self.cfg.max_P, 
            max_Q = self.cfg.max_Q, 
            max_D = self.cfg.max_D, 
            error_action = 'ignore', 
            suppress_warnings = True, 
            )
        return self 
    
    def predict(self, horizon:int) -> np.ndarray:
        return self.model.predict(n_periods=horizon) 

def arima_predict(Y_df:pd.DataFrame,
                  horizon: int = 1, 
                  n_jobs: int = 1, 
                  arima_builder=lambda:ArimaForecaster(ARIMAConfig(seasonal=False, m=1))):
    """
    Y_df: (N, Target) DataFrame of targets (e.g., your derived 400+ log-return targets), index=date_id-sorted.
    Returns: np.ndarray of shape (Target,) with 1-step predictions at the end of Y_df.
    """
    print(" --- Model: ARIMA")
    t_s = time() 
    cols = list(Y_df.columns)
    last_vals = [] 

    def _one(col):
        y = Y_df[col].to_numpy(dtype=float)
        try:
            model = arima_builder().fit(y) 
            return model.predict(horizon)[0] 
        except Exception as e:
            warnings.warn(f"ARIMA failed on {col}: {e}") 
            return np.nan 
    
    if n_jobs == 1:
        for c in cols: last_vals.append(_one(c)) 

    else:
        last_vals = Parallel(n_jobs=n_jobs, backend='loky')(delayed(_one)(c) for c in cols)
    #print(f"Prediction --- done!  |  Duration: {time()-t_s}") 
    return np.array(last_vals, dtype=float)  # shape (Target,)

# Ensemble
def weighted_ensemble(lstm_pred, arima_pred, w):
    """
    lstm_pred, arima_pred: (..., TGT) arrays
    w: scalar, shape (TGT,), or (..., TGT) broadcastable
    returns same shape as inputs
    """
    return w * lstm_pred + (1.0 - w) * arima_pred

def tune_weight_global(y_true_mat, y_lstm_mat, y_arima_mat, grid=np.linspace(0,1,51)):
    """
    y_*_mat: (W, TGT) matrices over a small validation window of length W.
    Returns scalar w in [0,1] minimizing mean MSE over all targets.
    """
    best_w, best_m = 0.5, np.inf
    for w in grid:
        m = mean_squared_error(y_true_mat, w*y_lstm_mat + (1-w)*y_arima_mat)
        if m < best_m: best_m, best_w = m, w
    return float(best_w)

def tune_weight_per_target(y_true_mat, y_lstm_mat, y_arima_mat, grid=np.linspace(0,1,21)):
    """
    Returns w_k for each target (vector length TGT).
    """
    TGT = y_true_mat.shape[1]
    ws = np.full(TGT, 0.5, dtype=float)
    for k in range(TGT):
        yt, yl, ya = y_true_mat[:,k], y_lstm_mat[:,k], y_arima_mat[:,k]
        m_best, w_best = np.inf, 0.5
        for w in grid:
            m = mean_squared_error(yt, w*yl + (1-w)*ya)
            if m < m_best: m_best, w_best = m, w
        ws[k] = w_best
    return ws

# Orchestration 
@dataclass 
class RunConfig:
    input_len: int = 64
    horizon: int = 1 #One-step ahead for labels 
    lstm_enable: bool = True 
    arima_enable: bool = True 
    ensemble_enable: bool = True 
    save_dir: str = "outputs"
    max_targets: Optional[int] = None # run a subset for spped; None=all.
    n_jobs: int = 1 # Parallelism over targets 



######################
# CNN + LSTM model 
#####################    
class CNNLSTMForecaster(nn.Module):
    """
    Conv1d over time that treats features as channels, followed by LSTM and Linear head.

    """
    def __init__(self, 
                 input_size: int, 
                 num_targets: int, # num_targets = output_size
                 conv_channels: int=128, 
                 kernel_size: int = 5, 
                 num_conv_layers: int = 2, 
                 lstm_hidden: int = 256, 
                 lstm_layers: int = 1, 
                 dropout: float = 0.1, 
                 output_len: int = 1):
        super().__init__()
        self.model_type = 'CNNLSTM'
        self.num_targets = num_targets
        assert kernel_size % 2 == 1, "Use an odd kernel_size to keep length with same padding"

        convs = [] 
        in_ch = input_size 
        for i in range(num_conv_layers):
            convs += [
                nn.Conv1d(
                    in_channels= in_ch, 
                    out_channels=conv_channels, 
                    kernel_size= kernel_size, 
                    padding = (kernel_size-1)//2, 
                    bias = True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) 
            ]
            in_ch = conv_channels 
        self.conv = nn.Sequential(*convs) 
        self.lstm = nn.LSTM(
            input_size = conv_channels, 
            hidden_size = lstm_hidden, 
            num_layers = lstm_layers, 
            batch_first = True, 
            dropout = dropout if lstm_layers > 1 else 0.0 
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), 
            nn.Linear(lstm_hidden, num_targets * output_len) 
        )
        self.output_len = output_len
        

    def forward(self, x): # x: (B, T, F)
        # CNN: Conv over time: want (B, C_in=F, L=T)
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.conv(x)       # (B, Cc, T)
        x = x.transpose(1, 2)  # (B, T, Cc)

        # LSTM over time 
        out, _ = self.lstm(x)  # (B, T, H) 

        # One-step-ahead (use last timestep)
        last = out[:, -1, :]  # (B, H)
        y = self.head(last)   # (B, Targets)

        # Match your LSTM forcaster API: (B, output_len, Targets)
        return y.view(-1, self.output_len, self.num_targets) #(Batch, Horizon, Targets)

def make_windows(Y: np.ndarray, input_len: int, X: Optional=None):
    """ 
    Build sliding windows:
    - If X is provided (multi-variate), windows from X; target= next Y row
    - If X is None (univariate AR), use Y as the single feature channel. 
    Returns tensors: Xw (N, T, F), Yw (N, Targets) 
        where... 
        N = number of sequences (batches, windows)
        T = timesteps per sequence (your input_len, ie., 88) 
        F = the number of features per timestep (i.e., 400) 
    """
    if X is None:
        X = Y  # if you’re using multivariate X, pass it explicitly

    T = len(Y)
    xs, ys = [], []
    for t in range(input_len, T):
        y_t = Y[t, :]
        # keep only if target vector is fully finite
        if not np.isfinite(y_t).all():
            continue
        xs.append(X[t-input_len:t, :])
        ys.append(y_t)

    if len(xs) == 0:
        raise ValueError("No finite target windows. Check Y construction & lags.")

    Xw = np.stack(xs, axis=0).astype(np.float32)  # (N, T, F)
    Yw = np.stack(ys, axis=0).astype(np.float32)  # (N, Targets)
    return Xw, Yw


def build_cnnlstm(cfg, num_features, num_targets):
    model = CNNLSTMForecaster(
        input_size=num_features,
        hidden_size=cfg.hidden_size,
        num_layers=1,
        dropout=cfg.dropout,
        output_len=1,
        num_targets=num_targets,
        cnn_kernel=getattr(cfg,"cnn_kernel",3),
        cnn_dilation=getattr(cfg,"cnn_dilation",1),
    )
    x_scaler = StandardScaler2D()
    return model, x_scaler

######################
# Prophet model 
######################



######################
# TCN(Temporal Convolutional Network) + residual blocks  
######################


class Chomp1d(nn.Module):
    def __init__(self, chomp_size): 
        super().__init__()
        self.chomp_size = chomp_size 
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size>0 else x 


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        # Use padding='same' to preserve time dimension
        self.conv1 = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding='same', dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding='same', dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1, padding='same') if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNForecaster(nn.Module):
    def __init__(self, input_size: int, 
                 num_targets: int, 
                 channels_list: list, 
                 kernel_size: int, 
                 dropout: float,
                 output_len: int = 1):
        super().__init__()
        self.model_type = 'TCN'
        self.input_size = input_size
        self.num_targets = num_targets
        self.output_len = output_len 
        layers = []
        num_levels = len(channels_list)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else channels_list[i-1]
            out_channels = channels_list[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(channels_list[-1], num_targets * self.output_len)

    def forward(self, x):
        # x: (B, T, F) -> transpose to (B, F, T) for Conv1d
        x = x.transpose(1, 2)  # (B, F, T)
        y = self.tcn(x)  # (B, channels_list[-1], T)
        y = y[:, :, -1]  # Take last time step: (B, channels_list[-1])
        y = self.head(y)  # (B, num_targets * output_len)
        return y.view(-1, self.output_len, self.num_targets)  # (B, output_len, num_targets)


def build_tcn(cfg, num_features, num_targets):
    channels = [cfg.channels]*cfg.blocks
    model = TCNForecaster(
        input_size=num_features,
        num_targets=num_targets,
        channels_list=channels,
        kernel_size=getattr(cfg,"tcn_kernel",3),
        dropout=cfg.dropout, 
        output_len = cfg.output_len 
    )
    x_scaler = StandardScaler2D()
    return model, x_scaler

######################
# Transformer
######################
class PositionalEncoding(nn.Module):
    """
    Adds positional information to the input embeddings to help the Transformer 
    distinguish between different time steps in the sequence. This is crucial 
    for time series data where order matters. The encoding is fixed and based 
    on sine/cosine functions of different frequencies.
    
    Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
             PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model (int): The dimension of the model embeddings.
        max_len (int): The maximum sequence length to precompute (default: 5000).
        dropout (float): Dropout rate after adding positional encoding.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Precompute positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension: (1, max_len, d_model)
        self.register_buffer('pe', pe)  # Not a trainable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) - Add positional encoding to the input
        x = x + self.pe[:, :x.size(1), :]  # Slice pe to match sequence length T
        return self.dropout(x)



class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the Transformer encoder, consisting of:
    1. Multi-head self-attention mechanism.
    2. Feed-forward network (two linear layers with ReLU).
    3. Layer normalization and residual connections around each sub-layer.
    
    This layer processes the sequence while attending to all positions simultaneously.
    
    Args:
        d_model (int): Embedding dimension.
        nhead (int): Number of attention heads.
        dim_feedforward (int): Hidden dimension of the feed-forward network.
        dropout (float): Dropout rate.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention sub-layer with residual connection and normalization
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout(attn_output))
        
        # Feed-forward sub-layer with residual connection and normalization
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_output))
        
        return src


class TRANSforecaster(nn.Module):
    """
    Transformer-based forecaster for time series data.
    
    Architecture Overview:
    - Input Projection: Linear layer to project input features (F) to d_model.
    - Positional Encoding: Adds time-step information to the embeddings.
    - Encoder Layers: Stack of TransformerEncoderLayers for sequence encoding.
    - Pooling: Global average pooling over time dimension to summarize the sequence.
    - Head: Linear layer to produce forecasts for output_len steps and num_targets.
    
    This is an encoder-only Transformer, suitable for forecasting where the entire 
    input sequence is encoded, then pooled, and mapped to predictions. For multi-step 
    forecasting (output_len > 1), the head outputs a flattened vector reshaped to 
    (B, output_len, num_targets). If you need autoregressive decoding or more advanced 
    features (e.g., cross-attention with decoder), you can extend this by adding a 
    decoder component.
    
    Unified support:
    - Univariate: input_size=1, num_targets=1
    - Multivariate: input_size=F, num_targets=T
    
    Args:
        input_size (int): Number of input features (F).
        d_model (int): Embedding dimension (hidden size for Transformer).
        nhead (int): Number of attention heads.
        num_layers (int): Number of encoder layers.
        dim_feedforward (int): Feed-forward hidden dimension (usually 4 * d_model).
        dropout (float): Dropout rate.
        output_len (int): Number of future steps to predict.
        num_targets (int): Number of target variables.
        max_len (int): Max sequence length for positional encoding.
    """
    def __init__(self, 
                 input_size: int, 
                 d_model: int, 
                 nhead: int = 8, 
                 num_layers: int = 2, 
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1, 
                 output_len: int = 1, 
                 num_targets: int = 1, 
                 max_len: int = 5000):
        super().__init__()
        self.model_type = 'TRANSFORMER'
        self.output_len = output_len
        self.num_targets = num_targets
        
        # Project input features to d_model dimension
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Stack of Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Global average pooling over time (T) to get a fixed-size representation
        self.pool = nn.AdaptiveAvgPool1d(1)  # Reduces (B, T, d_model) to (B, d_model, 1)
        
        # Linear head to map pooled features to (output_len * num_targets)
        self.head = nn.Linear(d_model, output_len * num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) - Input sequence
        
        # Project features to d_model: (B, T, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding: (B, T, d_model)
        x = self.pos_encoder(x)
        
        # Transpose for MultiheadAttention: (T, B, d_model) as it expects seq_len first
        x = x.transpose(0, 1)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)  # (T, B, d_model)
        
        # Transpose back: (B, T, d_model)
        x = x.transpose(0, 1)
        
        # Global average pooling over time: (B, d_model, 1) -> squeeze to (B, d_model)
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        
        # Linear head: (B, output_len * num_targets)
        y = self.head(pooled)
        
        # Reshape to match API: (B, output_len, num_targets)
        return y.view(-1, self.output_len, self.num_targets)

def build_trans(cfg, num_features, num_targets):
    """
    Builder function for TRANSforecaster, similar to build_cnnlstm or build_tcn.
    Uses cfg.hidden_size as d_model, and allows tweaking via cfg attributes.
    """
    model = TRANSforecaster(
        input_size=num_features,
        d_model=cfg.hidden_size,
        nhead=getattr(cfg, "nhead", 8),  # Number of attention heads (tweakable)
        num_layers=cfg.num_layers,
        dim_feedforward=getattr(cfg, "dim_feedforward", cfg.hidden_size * 4),  # FF hidden dim
        dropout=cfg.dropout,
        output_len= cfg.output_len,  # Typically 1, but can be set higher
        num_targets=num_targets,
        max_len=getattr(cfg, "max_len", 5000)  # Max sequence length for PE
    )
    x_scaler = StandardScaler2D()
    return model, x_scaler



######################
# FEDformer model 
######################
class FEDForecaster(nn.Module):
    """
    Simplified FEDformer-inspired forecaster.
    
    Key Components (inspired by FEDformer):
    - Decomposition: Splits input into trend and seasonal using SeriesDecomposer.
    - Encoder: Transformer layers with interspersed FrequencyEnhancedBlocks for freq-domain attention.
    - Positional Encoding: Standard PE.
    - Pooling & Head: Average pool over time, then linear to predictions.
    
    This is encoder-only for simplicity, focusing on frequency enhancement and decomposition.
    For full FEDformer, you'd add a decoder with auto-correlation, but this is tweakable starting point.
    Supports univariate/multivariate like your other models.
    
    Args:
        input_size (int): Input features (F).
        d_model (int): Embedding dimension.
        nhead (int): Attention heads.
        num_layers (int): Encoder layers.
        dim_feedforward (int): FF hidden dim.
        dropout (float): Dropout.
        output_len (int): Forecast steps.
        num_targets (int): Targets.
        decom_kernel_size (int): Decomposer window.
        top_k_modes (int): Top frequencies in FEB.
        max_len (int): Max seq len for PE.
    """
    def __init__(self, 
                 input_size: int, 
                 d_model: int, 
                 nhead: int = 8, 
                 num_layers: int = 2, 
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1, 
                 output_len: int = 1, 
                 num_targets: int = 1, 
                 decom_kernel_size: int = 25,
                 top_k_modes: int = 5,
                 max_len: int = 5000):
        super().__init__()
        self.model_type = 'FED'
        self.output_len = output_len    
        self.num_targets = num_targets  # num_targets = output_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Decomposition
        self.decomposer = SeriesDecomposer(kernel_size=decom_kernel_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder layers (alternate Transformer layer and Frequency block)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.freq_blocks = nn.ModuleList([
            FrequencyEnhancedBlock(d_model, top_k_modes, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooling and head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, output_len * num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        
        # Decompose
        trend, seasonal = self.decomposer(x)  # Both (B, T, F)
        x_decomp = trend + seasonal  # Recompose after processing, but here process seasonal with freq
        
        # Project
        x = self.input_projection(seasonal)  # Focus freq on seasonal (B, T, d_model)
        
        # Pos encoding
        x = self.pos_encoder(x)
        
        # Transpose for attn (T, B, d_model)
        x = x.transpose(0, 1)
        
        # Encoder: Alternate standard Transformer layer and freq block
        for enc_layer, freq_block in zip(self.encoder_layers, self.freq_blocks):
            x = enc_layer(x)  # Standard attn (T, B, d_model)
            x = x.transpose(0, 1)  # (B, T, d_model) for freq block
            x = freq_block(x)  # Freq enhanced
            x = x.transpose(0, 1)  # Back to (T, B, d_model)
        
        # Transpose back (B, T, d_model)
        x = x.transpose(0, 1)
        
        # Add back trend (simple fusion; could be more sophisticated)
        trend_proj = self.input_projection(trend)  # (B, T, d_model)
        x = x + trend_proj
        
        # Pool and head
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        y = self.head(pooled)  # (B, output_len * num_targets)
        return y.view(-1, self.output_len, self.num_targets)



# Helper for Series Decomposition (inspired by FEDformer's moving average decomposer)
class SeriesDecomposer(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size % 2 == 0:
            print(f"Error - You have put decom_kernel_size:{kernel_size}")
            print(f"--- Please modify decom_kernel_size in study.py")
            raise ValueError(f"kernel_size must be odd for symmetric decomposition, got {kernel_size}")
        self.kernel_size = kernel_size
        # Ensure padding maintains the sequence length
        self.padding = kernel_size - 1  # Total padding to account for kernel size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=self.padding // 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input shape: (B, T, F)"""
        x_trans = x.transpose(1, 2)  # (B, F, T)
        # Apply padding to maintain sequence length
        if self.padding % 2 == 1:
            # For odd kernel_size, pad asymmetrically to ensure exact length
            x_trans = torch.nn.functional.pad(x_trans, (self.padding // 2, self.padding // 2 + 1))
        else:
            x_trans = torch.nn.functional.pad(x_trans, (self.padding // 2, self.padding // 2))
        trend_trans = self.avg(x_trans)  # (B, F, T)
        seasonal_trans = x_trans - trend_trans  # (B, F, T)
        trend = trend_trans.transpose(1, 2)  # (B, T, F)
        seasonal = seasonal_trans.transpose(1, 2)  # (B, T, F)
        return trend, seasonal



# Frequency Enhanced Attention Block (simplified from FEDformer)
class FrequencyEnhancedBlock(nn.Module):
    """
    Performs attention in frequency domain using FFT.
    - Transforms time series to frequency domain.
    - Selects top-k dominant frequencies (modes) for sparse attention.
    - Inverse FFT back to time domain.
    
    This reduces complexity and captures periodic patterns better than time-domain attention.
    
    Args:
        d_model (int): Feature dimension.
        top_k (int): Number of top frequencies to keep (sparsity control).
        dropout (float): Dropout rate.
    """
    def __init__(self, d_model: int, top_k: int = 5, dropout: float = 0.1):
        super().__init__()
        self.top_k = top_k
        self.dropout = nn.Dropout(dropout)
        # Linear layers for query/key/value in frequency space (optional, but helps)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        B, T, D = x.shape
        
        # FFT to frequency domain: Real FFT (rfft) for real-valued inputs
        x_freq = torch.fft.rfft(x, dim=1)  # (B, T//2 + 1, D) complex
        
        # Amplitude for mode selection
        amp = torch.abs(x_freq)  # (B, freq, D)
        
        # Select top-k frequencies per batch/feature (mean over D for simplicity)
        amp_mean = amp.mean(dim=-1)  # (B, freq)
        _, top_indices = torch.topk(amp_mean, self.top_k, dim=1)  # (B, top_k)
        
        # Gather top modes (complex values)
        top_freq = torch.gather(x_freq, 1, top_indices.unsqueeze(-1).expand(-1, -1, D))  # (B, top_k, D)
        
        # Simple attention in freq domain: Project Q/K/V
        Q = self.q_proj(top_freq.real)  # Use real part for simplicity; could use complex ops
        K = self.k_proj(top_freq.real)
        V = self.v_proj(top_freq.real)
        
        # Attention scores (softmax over freq dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(D)  # (B, top_k, top_k)
        attn = torch.softmax(scores, dim=-1)  # (B, top_k, top_k)
        attn_output = torch.matmul(attn, V)  # (B, top_k, D)
        
        # Scatter back to full freq (zero elsewhere)
        x_freq_enhanced = torch.zeros_like(x_freq)
        x_freq_enhanced.scatter_(1, top_indices.unsqueeze(-1).expand(-1, -1, D), attn_output + 1j * attn_output)  # Approx complex
        
        # Inverse FFT to time domain
        x_time = torch.fft.irfft(x_freq_enhanced, n=T, dim=1)  # (B, T, D)
        
        return self.dropout(x_time)



class FEDForecaster(nn.Module):
    """
    Simplified FEDformer-inspired forecaster.
    
    Key Components (inspired by FEDformer):
    - Decomposition: Splits input into trend and seasonal using SeriesDecomposer.
    - Encoder: Transformer layers with interspersed FrequencyEnhancedBlocks for freq-domain attention.
    - Positional Encoding: Standard PE.
    - Pooling & Head: Average pool over time, then linear to predictions.
    
    This is encoder-only for simplicity, focusing on frequency enhancement and decomposition.
    For full FEDformer, you'd add a decoder with auto-correlation, but this is tweakable starting point.
    Supports univariate/multivariate like your other models.
    
    Args:
        input_size (int): Input features (F).
        d_model (int): Embedding dimension.
        nhead (int): Attention heads.
        num_layers (int): Encoder layers.
        dim_feedforward (int): FF hidden dim.
        dropout (float): Dropout.
        output_len (int): Forecast steps.
        num_targets (int): Targets.
        decom_kernel_size (int): Decomposer window.
        top_k_modes (int): Top frequencies in FEB.
        max_len (int): Max seq len for PE.
    """
    def __init__(self, 
                 input_size: int, 
                 d_model: int, 
                 nhead: int = 8, 
                 num_layers: int = 2, 
                 dim_feedforward: int = 512, 
                 dropout: float = 0.1, 
                 output_len: int = 1, 
                 num_targets: int = 1, 
                 decom_kernel_size: int = 25,
                 top_k_modes: int = 5,
                 max_len: int = 5000):
        super().__init__()
        self.model_type = 'FED'
        self.output_len = output_len
        self.num_targets = num_targets
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Decomposition
        self.decomposer = SeriesDecomposer(kernel_size=decom_kernel_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder layers (alternate Transformer layer and Frequency block)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.freq_blocks = nn.ModuleList([
            FrequencyEnhancedBlock(d_model, top_k_modes, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooling and head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, output_len * num_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        
        # Decompose
        trend, seasonal = self.decomposer(x)  # Both (B, T, F)
        x_decomp = trend + seasonal  # Recompose after processing, but here process seasonal with freq
        
        # Project
        x = self.input_projection(seasonal)  # Focus freq on seasonal (B, T, d_model)
        
        # Pos encoding
        x = self.pos_encoder(x)
        
        # Transpose for attn (T, B, d_model)
        x = x.transpose(0, 1)
        
        # Encoder: Alternate standard Transformer layer and freq block
        for enc_layer, freq_block in zip(self.encoder_layers, self.freq_blocks):
            x = enc_layer(x)  # Standard attn (T, B, d_model)
            x = x.transpose(0, 1)  # (B, T, d_model) for freq block
            x = freq_block(x)  # Freq enhanced
            x = x.transpose(0, 1)  # Back to (T, B, d_model)
        
        # Transpose back (B, T, d_model)
        x = x.transpose(0, 1)
        
        # Add back trend (simple fusion; could be more sophisticated)
        trend_proj = self.input_projection(trend)  # (B, T, d_model)
        x = x + trend_proj
        
        # Pool and head
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, d_model)
        y = self.head(pooled)  # (B, output_len * num_targets)
        return y.view(-1, self.output_len, self.num_targets)


def build_fed(cfg, num_features, num_targets):
    """
    Builder for FEDForecaster, compatible with your other builders.
    Uses cfg.hidden_size as d_model. Tweakable params via cfg (e.g., add cfg.top_k_modes).
    """
    model = FEDForecaster(
        input_size=num_features,
        d_model=cfg.hidden_size,
        nhead=getattr(cfg, "nhead", 8),
        num_layers=cfg.num_layers,
        dim_feedforward=getattr(cfg, "dim_feedforward", cfg.hidden_size * 4),
        dropout=cfg.dropout,
        output_len= cfg.output_len ,  # Default to 1
        num_targets=num_targets,
        decom_kernel_size=getattr(cfg, "decom_kernel_size", 25),
        top_k_modes=getattr(cfg, "top_k_modes", 5),
        max_len=getattr(cfg, "max_len", 5000)
    )
    x_scaler = StandardScaler2D()
    return model, x_scaler



# =======================================================================
# TimesFM additions (model loader / wrapper)
# -----------------------------------------------------------------------
# We keep this minimal and defensive: import TimesFM from the installed
# package. If import fails, we raise a clear error with instructions.
# =======================================================================
import torch
import logging
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
class TimesFMNotFoundError(ImportError):
    pass

def load_timesfm_heads_bank(bank_path: str,
                            num_targets: int,
                            horizon_len: int,
                            hidden: int = 64,
                            dropout: float = 0.0,
                            device: str = "cpu") -> Dict[str, nn.Module]:
    """
    This function is for interence.
    Build a ModuleDict of heads and load the packed state_dict.
    Returns a dict-like (ModuleDict) with keys 't0'..'t{num_targets-1}'.
    """
    bank = nn.ModuleDict({f"t{j}": make_timesfm_head(horizon_len, hidden, dropout)
                          for j in range(num_targets)})
    state = torch.load(bank_path, map_location=device)
    bank.load_state_dict(state)
    bank.to(device)
    bank.eval()
    return bank


def load_timesfm_model(model_name: str = "google/timesfm-1.0-200m-pytorch",
                       device: str | None = None,
                       horizon_len: int = 128,
                       context_len: int | None = None,
                       batch_size: int = 32,
                       local_ckpt:str = '../checkpoints/timesfm-1.0-200m/torch_model.ckpt'):
    """
    Load TimesFM and return a PyTorch nn.Module suitable for finetuning.
    Supports:
      A) timesfm==1.3.x (your 'TimesFmTorch' class from timesfm_torch.py)
      B) newer TimesFm(...).torch_model API
    """
    if device is None:
        device = "cuda" if (torch.cuda.is_available()) else "cpu"

    # -------- Path B: TimesFm(...).torch_model (if available) --------
    try:
        import timesfm as t
        backend = "gpu" if ("cuda" in device and torch.cuda.is_available()) else "cpu"
        hp_kwargs = dict(backend=backend, per_core_batch_size=int(batch_size), horizon_len=int(horizon_len))
        if context_len is not None:
            hp_kwargs["context_len"] = int(context_len)
        tfm = t.TimesFm(
            hparams=t.TimesFmHparams(**hp_kwargs),
            checkpoint=t.TimesFmCheckpoint(huggingface_repo_id=model_name),
        )
        core = getattr(tfm, "torch_model", None)
        if isinstance(core, nn.Module):
            core.to(device).train()
            for p in core.parameters(): p.requires_grad = True
            assert sum(p.numel() for p in core.parameters()) > 0, "TimesFM torch_model has 0 params."
            return core
    except Exception:
        pass  # fall through to the TimesFmTorch path

    # -------- Path A: your v1.3.x 'TimesFmTorch' (timesfm_torch.py) --------
    try:
        # Try common module paths
        try:
            from timesfm.torch.timesfm_torch import TimesFmTorch
        except Exception:
            from timesfm.timesfm_torch import TimesFmTorch  # your layout

        from timesfm import timesfm_base as base

        # Build the object and explicitly load the checkpoint
        hp = base.TimesFmHparams(
            backend="gpu" if ("cuda" in device and torch.cuda.is_available()) else "cpu",
            per_core_batch_size=int(batch_size),
            horizon_len=int(horizon_len),
            **({"context_len": int(context_len)} if context_len is not None else {})
        )
        if local_ckpt:
            ckpt = base.TimesFmCheckpoint(path=local_ckpt)
        else:
            ckpt = base.TimesFmCheckpoint(huggingface_repo_id=model_name)
        tfm_torch = TimesFmTorch(hparams=hp, checkpoint=ckpt)
        tfm_torch.load_from_checkpoint(ckpt)  # <- crucial for v1.3.x

        core = getattr(tfm_torch, "_model", None)  # PatchedTimeSeriesDecoder
        if not isinstance(core, nn.Module):
            raise RuntimeError("TimesFmTorch._model not set after load_from_checkpoint().")

        core.to(device).train()
        for p in core.parameters(): p.requires_grad = True
        assert sum(p.numel() for p in core.parameters()) > 0, "TimesFmTorch core has 0 params."
        return core

    except Exception as e:
        raise RuntimeError(
            "Failed to load a trainable TimesFM model. "
            "Checked both TimesFm(...).torch_model and TimesFmTorch paths."
        ) from e


class TimesFMPerTargetHead(nn.Module):
    """
    Tiny MLP head mapping [B, H] -> [B, H].
    Default: residual 2-layer MLP; cheap and expressive.
    """
    def __init__(self, horizon_len: int, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(horizon_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon_len),
        )

    def forward(self, x):                 # x: [B, H]
        return x + self.net(x)            # residual calibration


class TimesFMHeadWrapper(nn.Module):
    """
    Wraps a TimesFM torch model and swaps the mean-channel ([..., 0]) of patch 0
    with the output of a small per-target head.
    - base:  returns [B, P, H, 1+Q]
    - head:  takes [B, H] and returns [B, H]
    """
    def __init__(self, base: nn.Module, head: nn.Module | None, freeze_base: bool = True):
        super().__init__()
        self.base = base
        self.head = head
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x_context, x_padding, freq):
        pred = self.base(x_context, x_padding, freq)      # [B, P, H, 1+Q]
        if self.head is None:
            return pred
        mean = pred[..., 0]                               # [B, P, H]
        # calibrate first patch only (what your finetuner uses)
        base_feat = mean[:, 0, :]                         # [B, H]
        adj = self.head(base_feat)                        # [B, H]
        mean = mean.clone()
        mean[:, 0, :] = adj
        pred = pred.clone()
        pred[..., 0] = mean
        return pred


def make_timesfm_head(horizon_len: int, hidden: int = 64, dropout: float = 0.0):
    return TimesFMPerTargetHead(horizon_len, hidden, dropout)



####################################
# Informer 
####################################
class ProbSparseAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=10, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbSparseAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_Q, E = Q.shape
        _, _, L_K, _ = K.shape
        device = Q.device  # <<< add

        # sample_k indices in [0, min(L_Q, L_K))
        index_sample = torch.randint(
            low=0, high=min(L_Q, L_K),
            size=(L_Q, sample_k),
            device=device, dtype=torch.long,  # <<< add device & dtype
        )

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # arange must be on device and long
        arange_LQ = torch.arange(L_Q, device=device, dtype=torch.long).unsqueeze(1)  # <<< add

        K_sample = K_expand[:, :, arange_LQ, index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # topk indices already on device, but make dtype explicit
        M_top = M.topk(n_top, sorted=False)[1].to(dtype=torch.long)

        # build batch/head indexers on the right device
        b_idx = torch.arange(B, device=device, dtype=torch.long)[:, None, None]
        h_idx = torch.arange(H, device=device, dtype=torch.long)[None, :, None]

        Q_reduce = Q[b_idx, h_idx, M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top


    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        logging.debug(f"_get_initial_context: V shape={V.shape}, L_Q={L_Q}")
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, D).contiguous()
        else:
            if L_Q != L_V:
                logging.warning(f"Sequence length mismatch: L_Q={L_Q}, L_V={L_V}. Adjusting L_V to L_Q.")
                if L_V < L_Q:
                    V = F.pad(V, (0, 0, 0, L_Q - L_V))
                elif L_V > L_Q:
                    V = V[:, :, :L_Q, :]
                L_V = L_Q
            contex = V.cumsum(dim=-2).contiguous()
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        device = context_in.device  # <<< add
        index = index.to(device=device, dtype=torch.long)  # <<< add

        if self.mask_flag:
            reduced_L_Q = scores.shape[2]
            attn_mask = TriangularCausalMask(B, reduced_L_Q, device=device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        # Use actual length of context slot we're writing into
        L_slot = context_in.shape[2]                     # <<< add
        if torch.any(index >= L_slot) or torch.any(index < 0):
            # If this ever hits, something upstream is off; clamp as last resort
            index = index.clamp_(0, L_slot - 1)          # <<< add

        b_idx = torch.arange(B, device=device, dtype=torch.long)[:, None, None]   # <<< add
        h_idx = torch.arange(H, device=device, dtype=torch.long)[None, :, None]   # <<< add

        context_in[b_idx, h_idx, index, :] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_slot, L_slot], device=device) / L_slot).type_as(attn)
            attns[b_idx, h_idx, index, :] = attn
            return (context_in.contiguous(), attns)
        return (context_in.contiguous(), None)


    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape
        logging.debug(f"ProbSparseAttention forward: queries={queries.shape}, keys={keys.shape}, values={values.shape}")
        queries = queries.transpose(2, 1).contiguous()
        keys = keys.transpose(2, 1).contiguous()
        values = values.transpose(2, 1).contiguous()
        U_part = self.factor * np.ceil(np.log(min(L_Q, L_K))).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = min(U_part, L_K)
        u = min(u, L_Q)
        logging.debug(f"U_part={U_part}, u={u}")
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        context = self._get_initial_context(values, u)
        context, attn = self._update_context(context, values, scores_top, index, u, attn_mask)
        return context.contiguous().transpose(2, 1), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False, dropout=0.1):
        super().__init__()
        self.attention = attention
        self.n_heads = n_heads
        self.d_model = d_model
        self.mix = mix 

        # If not provided, default to d_model // n_heads (Informer convention)
        self.d_k = d_keys if d_keys is not None else d_model // n_heads
        self.d_v = d_values if d_values is not None else d_model // n_heads

        # Project into head spaces
        self.query_projection = nn.Linear(d_model, n_heads * self.d_k)
        self.key_projection   = nn.Linear(d_model, n_heads * self.d_k)
        self.value_projection = nn.Linear(d_model, n_heads * self.d_v)

        # IMPORTANT: in_features must be H * d_v (NOT hard-coded d_model)
        self.out_projection = nn.Linear(self.n_heads * self.d_v, self.d_model)

        self.dropout = nn.Dropout(dropout)
        logging.debug(f"AttentionLayer initialized: d_model={d_model}, n_heads={n_heads}, d_keys={d_keys}, d_values={d_values}")

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_q, _ = queries.shape
        B, L_k, _ = keys.shape
        B, L_v, _ = values.shape

        # project
        Q = self.query_projection(queries)  # (B, L_q, H*d_k)
        K = self.key_projection(keys)       # (B, L_k, H*d_k)
        V = self.value_projection(values)   # (B, L_v, H*d_v)

        # reshape to (B, L, H, d)
        Q = Q.view(B, L_q, self.n_heads, self.d_k)
        K = K.view(B, L_k, self.n_heads, self.d_k)
        V = V.view(B, L_v, self.n_heads, self.d_v)

        # route by attention kind
        if getattr(self.attention, "input_layout", None) == "blhe":
            # FullAttention path – already (B, L, H, d)
            out, attn = self.attention(Q, K, V, attn_mask=attn_mask)  # context: (B,L_q,H,d_v)
            # concat heads -> (B, L_q, H*d_v)
            out = out.contiguous().view(B, L_q, self.n_heads * self.d_v)

        else:
            # ProbSparse (expects (B, H, L, d))
            Q = Q.permute(0, 2, 1, 3).contiguous()  # (B,H,L_q,d_k)
            K = K.permute(0, 2, 1, 3).contiguous()  # (B,H,L_k,d_k)
            V = V.permute(0, 2, 1, 3).contiguous()  # (B,H,L_v,d_v)
            out, attn = self.attention(Q, K, V, attn_mask=attn_mask)   # (B,H,L_q,d_v)
            out = out.permute(0, 2, 1, 3).contiguous().view(B, L_q, self.n_heads * self.d_v)

        return self.out_projection(out), attn

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

class DecoderLayer(nn.Module):
    def __init__(self, self_attention: AttentionLayer,
                 cross_attention: AttentionLayer,
                 d_model: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) 
        self.norm3 = nn.LayerNorm(d_model) 
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu
        # guard: 
        assert self.self_attention.n_heads == self.cross_attention.n_heads, \
            f"Decoder self-attn heads ({self.self_attention.n_heads}) != cross-attn heads ({self.cross_attention.n_heads})"


    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=3, padding=2, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, factor=None):
        super().__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.input_layout = "blhe"  # << declare intent

    def forward(self, queries, keys, values, attn_mask=None):
        # queries: (B, L_q, H, E), keys: (B, S, H, E), values: (B, S, H, V)
        Bq, Lq, Hq, Eq = queries.shape
        Bk, Sk, Hk, Ek = keys.shape
        Bv, Sv, Hv, Vd = values.shape

        # Runtime guards (#4)
        assert Bq == Bk == Bv, f"Batch mismatch: {Bq}, {Bk}, {Bv}"
        assert Hq == Hk == Hv, f"Heads mismatch: Q={Hq}, K={Hk}, V={Hv}"
        assert Eq == Ek,       f"Key/query head dims mismatch: {Eq} vs {Ek}"
        assert Sk == Sv,       f"Key/Value length mismatch: {Sk} vs {Sv}"

        scale = self.scale or (Eq ** -0.5)
        scores = torch.einsum("blhe,bshe->blhs", queries, keys) * scale  # (B,Lq,H,S)

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        A = torch.softmax(scores, dim=-1)
        A = self.dropout(A)
        context = torch.einsum("blhs,bshe->blhe", A, values)  # (B,Lq,H,Vd)

        return (context, A) if self.output_attention else (context, None)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='d', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x)
        if x_mark is not None:
            logging.debug(f"DataEmbedding: x={x.shape}, x_mark={x_mark.shape}")
            x_mark = x_mark[:, :, :x.shape[-1]]
            x = x + x_mark
        return self.dropout(x)

class InForecaster(nn.Module):
    def __init__(self, input_size: int, enc_in: int, dec_in: int, c_out: int, 
                 seq_len: int, label_len: int, out_len: int, 
                 d_model=512, n_heads=8, d_k=None, d_v=None, e_layers=3, d_layers=2, d_ff=512, 
                 dropout=0.0, attn='prob', embed='fixed', freq='d', activation='gelu', 
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0')):
        super(InForecaster, self).__init__()
        self.model_type = 'Informer'
        self.attn = attn
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = out_len
        self.output_attention = output_attention
        self.num_targets = c_out  # output_size 
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_model // self.n_heads
        self.d_v = d_v if d_v is not None else d_model // self.n_heads
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        
        # Encoder
        Attn = ProbSparseAttention if attn == 'prob' else FullAttention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor=10, attention_dropout=dropout, output_attention=output_attention), 
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(d_model) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor=10, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    AttentionLayer(FullAttention(False, factor=10, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                ) for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        self.projection = nn.Linear(d_model, self.num_targets)
        self.to(device)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        logging.debug(f"InForecaster forward: x_enc={x_enc.shape}, x_mark_enc={x_mark_enc.shape if x_mark_enc is not None else None}, "
                      f"x_dec={x_dec.shape}, x_mark_dec={x_mark_dec.shape if x_mark_dec is not None else None}")
        if torch.isnan(x_enc).any() or (x_mark_enc is not None and torch.isnan(x_mark_enc).any()) or \
           torch.isnan(x_dec).any() or (x_mark_dec is not None and torch.isnan(x_mark_dec).any()):
            logging.warning("NaNs detected in input tensors")
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out[:, -self.pred_len:, :])
        #dec_out = dec_out.reshape(-1, self.pred_len, self.num_targets)
        
        if self.output_attention:
            return dec_out, attns
        return dec_out

    
# models/derits.py
from spectral import (
    apply_freq_derivative, inverse_freq_derivative,
    ComplexLinear, ComplexDepthwiseConvFreq, BandMask )

class FourierConvBlock(nn.Module):
    """
    Frequency-domain conv block:
      - 1x1 complex linear (channel mix)
      - depthwise conv over frequency bins
      - residual + dropout
    """
    def __init__(self, channels, kernel_size=5, dropout=0.1):
        super().__init__()
        self.cmix = ComplexLinear(channels, channels)
        self.dw = ComplexDepthwiseConvFreq(channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, zc):
        # zc: complex [B, C, F]
        res = zc
        zc = self.cmix(zc)
        zc = self.dw(zc)
        return res + self.dropout(zc)

class DerivativeBranch(nn.Module):
    """
    One derivative order branch:
      - Apply (i2πf)^k
      - Band mask (learnable)
      - N frequency conv blocks
      - Optional projection to D targets
      - Inverse derivative + iRFFT back to time
    """
    def __init__(self, order: int, L: int, C_in: int, D: int, depth=2, kernel_size=5, dropout=0.1):
        super().__init__()
        self.order = order
        self.L = L
        self.D = D
        self.proj_in = ComplexLinear(C_in, C_in)
        self.blocks = nn.ModuleList([
            FourierConvBlock(C_in, kernel_size=kernel_size, dropout=dropout) for _ in range(depth)
        ])
        self.band = None  # set at runtime based on F
        self.head = ComplexLinear(C_in, D)  # map channels to D targets in frequency space

    def _ensure_band(self, F, device, C_in):
        if self.band is None:
            self.band = BandMask(C_in, F).to(device)

    def forward(self, x):  # x: real tensor [B, L, C_in]
        B, L, C = x.shape

        with torch.amp.autocast("cuda", enabled=False):
            device = x.device
            # time -> freq (per channel)
            # transpose to [B, C, L] for rFFT along time
            xt = x.transpose(1, 2)  # [B, C, L]
            X = torch.fft.rfft(xt, dim=-1)  # [B, C, F] complex
            Fbins = X.shape[-1]
            self._ensure_band(Fbins, device, C)

            # frequency derivative
            Xd = apply_freq_derivative(X, order=self.order, L=L)  # complex [B,C,F]

            # pre-proj & band mask
            Xd = self.proj_in(Xd)
            Xd, mask = self.band(Xd)

            # OFCN stack
            for blk in self.blocks:
                Xd = blk(Xd)

            # project to D targets in freq
            Yd = self.head(Xd)  # [B, D, F] complex

            # invert derivative
            Yc = inverse_freq_derivative(Yd, order=self.order, L=L)  # [B, D, F] complex

            # back to time
            yt = torch.fft.irfft(Yc, n=L, dim=-1)  # [B, D, L] real
            y  = yt.transpose(1, 2).contiguous()  # [B, L, D] time-last

        # return last H positions are selected by the outer model head
        return y  # [B, L, D], per-branch contribution (time-domain)

class DeRiTSBackbone(nn.Module):
    """
    Multi-order branches + fusion + forecasting heads.
    Produces H-step forecasts for D targets.
    """
    def __init__(self, L: int, H: int, C_in: int, D: int, orders=(0,1,2), depth=2, kernel_size=5, dropout=0.1):
        super().__init__()
        self.L, self.H, self.C_in, self.D = L, H, C_in, D
        self.branches = nn.ModuleList([
            DerivativeBranch(order=o, L=L, C_in=C_in, D=D, depth=depth, kernel_size=kernel_size, dropout=dropout)
            for o in orders
        ])
        # fusion over branches (simple learnable weighted sum)
        self.fusion = nn.Parameter(torch.ones(len(orders)) / len(orders))
        # final temporal head over the last L samples -> H-step forecast
        self.temporal_head = nn.GRU(input_size=D, hidden_size=2*D, num_layers=1, batch_first=True)
        self.proj_out = nn.Linear(2*D, D)

    def forward(self, x):  # x: [B, L, C_in]
        B, L, C = x.shape
        assert L == self.L

        y_branches = []
        for b in self.branches:
            yb = b(x)  # [B, L, D]
            y_branches.append(yb)

        # fuse branches (time-domain)
        W = torch.softmax(self.fusion, dim=0)  # [nb]
        Y = 0.0
        for w, yb in zip(W, y_branches):
            Y = Y + w * yb  # [B, L, D]

        # take the last context and produce H-step forecasts via a small GRU head
        ctx = Y  # [B, L, D]
        out_seq, _ = self.temporal_head(ctx)  # [B, L, 2D]
        last = out_seq[:, -1:, :]  # [B,1,2D]
        # repeat or decode auto-regressively; here: simple MLP per step
        preds = []
        h = last
        for _ in range(self.H):
            step = self.proj_out(h.squeeze(1))  # [B, D]
            preds.append(step.unsqueeze(1))
            # optional: feed the predicted step back through a tiny GRU cell
            h, _ = self.temporal_head(step.unsqueeze(1))
        yhat = torch.cat(preds, dim=1)  # [B, H, D]
        return yhat

def build_derits(cfg, X_raw, Y_raw):
    """
    Builds a DeRiTSBackbone using fields from TrainConfig-like cfg.
    Expects:
      - cfg.input_len, cfg.output_len, cfg.input_size, cfg.output_size
      - cfg.orders (tuple/list), cfg.depth, cfg.kernel_size, cfg.dropout
    """
    L = int(cfg.model.params.get('input_len', cfg.input_len))
    H = int(cfg.model.params.get('output_len', cfg.output_len))
    C_in = int(cfg.model.params.get('input_size', cfg.input_size))
    D = int(cfg.model.params.get('output_size', cfg.output_size))
    orders = tuple(cfg.model.params.get('orders', (0,1,2)))
    depth = int(cfg.model.params.get('depth', 2))
    kernel_size = int(cfg.model.params.get('kernel_size', 5))
    dropout = float(cfg.model.params.get('dropout', cfg.dropout))
    rnn_hidden_mul = int(cfg.model.params.get('rnn_hidden_mul', 2))
    model = DeRiTSBackbone(L=L, H=H, C_in=C_in, D=D,
                           orders=orders, depth=depth, kernel_size=kernel_size,
                           dropout=dropout, rnn_hidden_mul=rnn_hidden_mul)
    return model



# Update MODEL_CLASSES
MODEL_CLASSES = {
    'LSTMForecaster': LSTMForecaster,
    'CNNLSTMForecaster': CNNLSTMForecaster,
    'TCNForecaster': TCNForecaster,
    'FEDForecaster': FEDForecaster,
    #'TimesFM': TimesFMHeadWrapper,
    'InForecaster': InForecaster,
    'DERITS': DeRiTSBackbone, #
}