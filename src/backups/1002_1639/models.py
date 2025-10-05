import pandas as pd 
import numpy as np 
import torch.nn as nn 
from torch.nn.utils import weight_norm 
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
    return out[0, 0, :], metadata  # (num_targets,), metadata

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
            nn.Linear(lstm_hidden, num_targets) 
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
        return y.unsqueeze(1) 

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
    def __init__(self, input_size: int, num_targets: int, channels_list: list, kernel_size: int, dropout: float):
        super().__init__()
        self.model_type = 'TCN'
        self.input_size = input_size
        self.num_targets = num_targets
        self.output_len = 1
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
        dropout=cfg.dropout
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
        output_len=1,  # Typically 1, but can be set higher
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
    """
    Decomposes time series into trend and seasonal components using a moving average kernel.
    - Trend: Smoothed version via avg pooling.
    - Seasonal: Input - Trend.
    
    Args:
        kernel_size (int): Size of the moving average window (odd recommended).
    """
    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (B, T, F), got shape {x.shape}")
        
        # Pad time dimension manually to avoid issues with torch.nn.functional.pad
        B, T, F = x.shape
        x_padded = torch.nn.functional.pad(
            x, (0, 0, self.padding, self.padding), mode='replicate'  # Pad time dim (T)
        )  # (B, T + 2*padding, F)
        
        # Apply moving average (avg pooling over time)
        x_padded = x_padded.transpose(1, 2)  # (B, F, T + 2*padding)
        trend = self.avg(x_padded).transpose(1, 2)  # (B, T, F)
        
        # Ensure trend matches input time length
        if trend.shape[1] != T:
            trend = trend[:, :T, :]  # Trim if necessary
        
        seasonal = x - trend  # (B, T, F)
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
        output_len=1,  # Default to 1
        num_targets=num_targets,
        decom_kernel_size=getattr(cfg, "decom_kernel_size", 25),
        top_k_modes=getattr(cfg, "top_k_modes", 5),
        max_len=getattr(cfg, "max_len", 5000)
    )
    x_scaler = StandardScaler2D()
    return model, x_scaler


# Update MODEL_CLASSES
MODEL_CLASSES = {
    'LSTMForecaster': LSTMForecaster,
    'CNNLSTMForecaster': CNNLSTMForecaster,
    'TCNForecaster': TCNForecaster,
    'FEDForecaster': FEDForecaster  # Add FEDForecaster
}
