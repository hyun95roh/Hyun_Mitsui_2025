import pandas as pd 
import numpy as np 
import os 
import polars as pl 
import warnings 
from dataclasses import dataclass 
from typing import Tuple, List, Dict, Callable, Optional, Protocol, Iterable
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from time import time 
import matplotlib.pyplot as plt
import yaml 
import logging
######################
# Submission  
######################
"""
def log_returns(data, lag):
    log_returns = pd.Series(np.nan, index=data.index)

    # compute log returns based on the rules
    for t in range(len(data)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') 
            try: 
                log_returns.iloc[t] = np.log(data.iloc[t + lag + 1] / data.iloc[t+1])
            except Exception:
                log_returns.iloc[t] = np.nan 
    return log_returns
    
def generate_targets(column_a: pd.Series, column_b: pd.Series, lag: int) -> pd.Series:
    return_a = log_returns(column_a, lag)
    return_b = log_returns(column_b, lag)

    return return_a - return_b 
"""

def log_returns(series:pd.Series, lag:int) -> pd.Series:
    """ 
    This function do the same as log_return in essence, but much faster. 
    - shift(-k) moves values backward, the computed return at row t is aligned with
    the same index as your loop would have assigned. 
    - At the tail of series, the shifts point past the en, so pandas fill with NaN.
    - At the front values are defined normally. 
    """
    s = series.astype(float)
    return np.log(s.shift(-(lag+1)) / s.shift(-1))

######################
# Ensemble predictions
######################
def weighted_ensemble(*predictions: np.ndarray, w: np.ndarray | float = None) -> np.ndarray:
    """
    Combine predictions from multiple models using weights.
    
    Args:
        *predictions: Variable number of prediction arrays, each of shape (num_targets,) or (1, num_targets).
        w: Weights for each model. Can be:
           - float: Single weight applied to first model, rest split equally.
           - np.ndarray: Array of shape (num_models,) for global weights or (num_targets, num_models) for per-target weights.
           - None: Equal weights (1/num_models).
    
    Returns:
        np.ndarray: Ensembled predictions of shape (num_targets,).
    
    Notes:
        - num_targets must be 424 for Mitsui challenge.
        - Predictions are clipped to finite values to avoid NaN/infinity issues.
        - If weights are invalid, falls back to equal weighting.
    """
    num_targets = 424  # Hard-coded for Mitsui challenge
    predictions = [np.asarray(p).reshape(-1) for p in predictions]  # Ensure 1D arrays
    
    # Validate predictions
    for i, pred in enumerate(predictions):
        if pred.shape != (num_targets,):
            logging.error(f"Prediction {i} has shape {pred.shape}, expected ({num_targets},)")
            raise ValueError(f"Prediction {i} has invalid shape {pred.shape}")
        if not np.isfinite(pred).all():
            logging.warning(f"Non-finite values in prediction {i}; replacing with zeros")
            predictions[i] = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)

    num_models = len(predictions)
    if num_models == 0:
        logging.error("No predictions provided")
        raise ValueError("At least one prediction array is required")

    # Handle weights
    if w is None:
        w = np.ones(num_models) / num_models
    elif isinstance(w, (int, float)):
        if num_models > 1:
            w = np.array([w] + [(1.0 - w) / (num_models - 1)] * (num_models - 1))
        else:
            w = np.array([1.0])
    else:
        w = np.asarray(w)
        if w.shape == (num_models,):
            pass  # Global weights
        elif w.shape == (num_targets, num_models):
            pass  # Per-target weights
        else:
            logging.warning(f"Invalid weight shape {w.shape}; using equal weights")
            w = np.ones(num_models) / num_models

    # Normalize weights
    if w.ndim == 1:
        w = w / np.sum(w)
        # Combine predictions: (num_targets,) = sum(w[i] * pred[i])
        final_pred = np.sum([w[i] * pred for i, pred in enumerate(predictions)], axis=0)
    else:
        w = w / np.sum(w, axis=1, keepdims=True)
        # Per-target weighting: (num_targets,) = sum(w[:,i] * pred[i])
        final_pred = np.sum([w[:, i:i+1] * pred for i, pred in enumerate(predictions)], axis=0)

    # Ensure finite output
    if not np.isfinite(final_pred).all():
        logging.warning("Non-finite values in final ensemble; replacing with zeros")
        final_pred = np.nan_to_num(final_pred, nan=0.0, posinf=0.0, neginf=0.0)

    logging.info(f"Ensembling {num_models} models with weights {w[:5] if w.ndim == 1 else w.shape}; output shape {final_pred.shape}")
    return final_pred



######################
# Config control 
######################
def validate_model_config(config_path: str) -> bool:
    """
    Validate a model config file for required fields and reasonable values.
    
    Args:
        config_path: Path to YAML file.
    
    Returns:
        bool: True if valid, raises ValueError if invalid.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_fields = ['model_name', 'model_class', 'params', 'optimizer', 'lr', 'epochs']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Config {config_path} missing required field: {field}")
    
    params = config['params']
    required_params = ['input_size', 'output_size', 'batch_size', 'hidden_size', 'num_layers', 'dropout']
    for param in required_params:
        if param not in params:
            raise ValueError(f"Config {config_path} missing param: {param}")
    
    # Additional checks for FED-specific params
    if config['model_class'] == 'FEDTrainer':
        fed_params = ['decom_kernel_size', 'top_k_modes', 'nhead', 'dim_feedforward']
        for param in fed_params:
            if param not in params:
                raise ValueError(f"FED config {config_path} missing param: {param}")
    
    return True

def validate_all_configs(config_dir: str):
    """Validate all .yaml files in config_dir."""
    for file in os.listdir(config_dir):
        if file.endswith('.yaml'):
            validate_model_config(os.path.join(config_dir, file))


######################
# loss functions
######################
import torch
import torch.nn.functional as F

def mae_loss(pred, target):
    return F.l1_loss(pred, target)

def pinball_loss(pred, target, q=0.5):
    e = target - pred
    return torch.maximum(q*e, (q-1)*e).mean()

def multi_quantile_loss(q_preds, target, quantiles=(0.1, 0.5, 0.9)):
    # q_preds: list of tensors [B,H,D] for each quantile
    loss = 0.0
    for p, q in zip(q_preds, quantiles):
        loss = loss + pinball_loss(p, target, q)
    return loss / len(quantiles)


def masked_mse(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.mean((y_true[m]-y_pred[m])**2)) if m.any() else np.nan

def concat_lag_labels_rowwise(
    lag_frames: Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame],
    expected_per_lag: Dict[int, int],
    cols_per_lag: Dict[int, List[str]]
) -> Tuple[np.ndarray, List[str]]:
    """
    Given 4 polars DataFrames for a SINGLE date_id (lag1..lag4),
    return a (num_labels,) numpy vector of ground-truth labels and their column names,
    after dropping ['date_id','label_date_id'] and concatenating horizontally in [lag1|lag2|lag3|lag4].
    
    If a lag_df has height == 0 (date_id not present), use np.full(expected_per_lag[lag], np.nan) for y_true_lag
    and cols_per_lag[lag] for columns.
    """
    cols_by_lag = []
    arrays = []
    for lag, lag_df in zip((1, 2, 3, 4), lag_frames):
        exp_n = expected_per_lag[lag]
        if lag_df.height == 0:
            arr = np.full(exp_n, np.nan, dtype=float)
            cols = cols_per_lag[lag]
        else:
            if lag_df.height != 1:
                raise ValueError(f"Expected exactly 1 row for date_id in lag {lag}, got {lag_df.height}")
            df = lag_df.drop(['date_id', 'label_date_id'])
            cols = df.columns
            arr = df.to_numpy().reshape(-1)
            if len(arr) != exp_n:
                raise ValueError(f"Unexpected number of columns for lag {lag}: {len(arr)} != {exp_n}")
        cols_by_lag.append(cols)
        arrays.append(arr)
    y_true = np.concatenate(arrays, axis=0)
    cols_flat = [c for cols in cols_by_lag for c in cols]
    return y_true, cols_flat

def concat_lag_labels_rowwise_safe(lag_frames):
    """
    lag_frames: (lag1_pl, lag2_pl, lag3_pl, lag4_pl) — 각 프레임은 해당 date_id 한 행만 포함.
    반환: (y_true_flat: np.ndarray, cols_flat: list[str])
    """
    arrays, cols = [], []
    for df in lag_frames:
        keep = [c for c in df.columns if c not in ("date_id","label_date_id")]
        if not keep:
            arrays.append(np.array([], dtype=float))
            continue
        arr = df.select(keep).to_pandas().to_numpy(dtype=float).reshape(-1)
        arrays.append(arr)
        cols.extend(keep)
    y_true = np.concatenate(arrays, axis=0) if arrays else np.array([], dtype=float)
    return y_true, cols

def per_lag_mse(y_true, y_pred, lag_dfs, lag):
    cols_per_lag = {lag_: df.shape[1] - 2 for lag_, df in lag_dfs.items()}
    mse = masked_mse(y_true, y_pred)

    # per-lag MSE using expected sizes (since y_true now full)
    n1 = cols_per_lag[1]
    n2 = cols_per_lag[2]
    n3 = cols_per_lag[3]
    n4 = cols_per_lag[4]
    s1, s2, s3 = 0, n1, n1 + n2
    temp = {1:(s1,s2), 2:(s2,s3), 3:(s3,s3+n3), 4:(s3+n3, s3+n3+n4)}
    a, b = temp[lag][0], temp[lag][1]
    if b > a:  # only if there are targets for this lag
        lag_mse = masked_mse(y_true[a:b], y_pred[a:b])

    return mse, lag_mse

def evaluate_mse_over_test(test_csv_path, lag_dir, predict_row_fn):
    test = pl.read_csv(test_csv_path)
    lag_files = {
        1: pl.read_csv(f"{lag_dir}/test_labels_lag_1.csv"),
        2: pl.read_csv(f"{lag_dir}/test_labels_lag_2.csv"),
        3: pl.read_csv(f"{lag_dir}/test_labels_lag_3.csv"),
        4: pl.read_csv(f"{lag_dir}/test_labels_lag_4.csv"),
    }

    # expected widths per lag (from headers)
    exp_cols = {lag: max(0, df.shape[1]-2) for lag, df in lag_files.items()}
    if sum(exp_cols.values()) == 0:
        return {
            "overall": np.nan,
            "per_lag": {lag: np.nan for lag in (1,2,3,4)},
            "by_lag_arrays": {lag: {"y_true": np.array([]), "y_pred": np.array([])} for lag in (1,2,3,4)}
        }

    # expected column names per lag (from headers)
    cols_per_lag = {}
    for lag, df in lag_files.items():
        cols_per_lag[lag] = [c for c in df.columns if c not in ("date_id","label_date_id")]

    date_ids = test['date_id'].unique(maintain_order=True).to_list()

    overall_se, overall_n = 0.0, 0
    perlag_se, perlag_n = {1:0.0,2:0.0,3:0.0,4:0.0}, {1:0,2:0,3:0,4:0}
    by_lag_arrays = {lag: {"y_true": [], "y_pred": []} for lag in (1,2,3,4)}

    for d in date_ids:
        tb = test.filter(pl.col('date_id')==d)
        b = {lag: lag_files[lag].filter(pl.col('date_id')==d) for lag in (1,2,3,4)}

        # (A) prediction vector (gateway-compatible): concatenated in [lag1|lag2|lag3|lag4] **using per-date columns**
        pred_row = predict_row_fn(tb, b[1], b[2], b[3], b[4], d)
        pred_row = pred_row.to_pandas() if isinstance(pred_row, pl.DataFrame) else pred_row
        y_pred_full = pred_row.to_numpy().reshape(-1)

        # (B) y_true_full using global expected widths (fill missing with NaN)
        y_true_parts = []
        per_date_cols = []  # actual per-date order for slicing
        for lag in (1,2,3,4):
            df = b[lag]
            if df.height == 0:
                y_true_parts.append(np.full(exp_cols[lag], np.nan, dtype=float))
                per_date_cols.extend(cols_per_lag[lag])
            else:
                keep = [c for c in df.columns if c not in ("date_id","label_date_id")]
                if len(keep) == 0:
                    y_true_parts.append(np.full(exp_cols[lag], np.nan, dtype=float))
                    per_date_cols.extend(cols_per_lag[lag])
                else:
                    arr = df.select(keep).to_pandas().to_numpy().reshape(-1)
                    # if per-date width < expected (it shouldn’t), pad; if > expected, trim
                    if len(arr) < exp_cols[lag]:
                        pad = np.full(exp_cols[lag]-len(arr), np.nan, dtype=float)
                        arr = np.concatenate([arr, pad])
                    elif len(arr) > exp_cols[lag]:
                        arr = arr[:exp_cols[lag]]
                    y_true_parts.append(arr)
                    per_date_cols.extend(keep)
        y_true_full = np.concatenate(y_true_parts, axis=0)

        # Defensive align lengths
        L = min(len(y_true_full), len(y_pred_full))
        if L == 0:
            continue
        y_true_full = y_true_full[:L]
        y_pred_full = y_pred_full[:L]

        m = np.isfinite(y_true_full) & np.isfinite(y_pred_full)
        if m.any():
            overall_se += float(np.sum((y_true_full[m]-y_pred_full[m])**2))
            overall_n  += int(np.sum(m))

        # per-lag
        offset = 0
        for lag in (1,2,3,4):
            k = exp_cols[lag]
            if k == 0:
                continue
            a, z = offset, offset + k
            yt, yp = y_true_full[a:z], y_pred_full[a:z]
            offset = z

            mlag = np.isfinite(yt) & np.isfinite(yp)
            if mlag.any():
                perlag_se[lag] += float(np.sum((yt[mlag]-yp[mlag])**2))
                perlag_n[lag]  += int(np.sum(mlag))
                by_lag_arrays[lag]["y_true"].append(yt[mlag])
                by_lag_arrays[lag]["y_pred"].append(yp[mlag])

    # finalize
    for lag in (1,2,3,4):
        if by_lag_arrays[lag]["y_true"]:
            by_lag_arrays[lag]["y_true"] = np.concatenate(by_lag_arrays[lag]["y_true"])
            by_lag_arrays[lag]["y_pred"] = np.concatenate(by_lag_arrays[lag]["y_pred"])
        else:
            by_lag_arrays[lag]["y_true"] = np.array([], dtype=float)
            by_lag_arrays[lag]["y_pred"] = np.array([], dtype=float)

    overall = (overall_se/overall_n) if overall_n>0 else np.nan
    per_lag = {lag: (perlag_se[lag]/perlag_n[lag]) if perlag_n[lag]>0 else np.nan for lag in (1,2,3,4)}
    return dict(overall=overall, per_lag=per_lag, by_lag_arrays=by_lag_arrays)

def evaluate_mse_over_test2(
    test_csv_path: str,
    lag_dir: str,
    predict_row_fn,
    return_arrays: bool = False
):
    test = pl.read_csv(test_csv_path)
    l1 = pl.read_csv(f"{lag_dir}/test_labels_lag_1.csv")
    l2 = pl.read_csv(f"{lag_dir}/test_labels_lag_2.csv")
    l3 = pl.read_csv(f"{lag_dir}/test_labels_lag_3.csv")
    l4 = pl.read_csv(f"{lag_dir}/test_labels_lag_4.csv")

    date_ids = test['date_id'].unique(maintain_order=True).to_list()

    mses = []
    mses_by_lag = {1: [], 2: [], 3: [], 4: []}

    # (optional) raw arrays for 디버깅/시각화
    by_lag_arrays = {1: {"y_true": [], "y_pred": []},
                     2: {"y_true": [], "y_pred": []},
                     3: {"y_true": [], "y_pred": []},
                     4: {"y_true": [], "y_pred": []}}

    for d in date_ids:
        tb = test.filter(pl.col('date_id')==d)
        b1 = l1.filter(pl.col('date_id')==d)
        b2 = l2.filter(pl.col('date_id')==d)
        b3 = l3.filter(pl.col('date_id')==d)
        b4 = l4.filter(pl.col('date_id')==d)

        # --- y_true (lag-safe)
        y_true_flat, cols_flat = concat_lag_labels_rowwise_safe((b1,b2,b3,b4))

        # --- 예측 (게이트웨이 호환 1×N DataFrame)
        pred_row = predict_row_fn(tb,b1,b2,b3,b4,d)
        if isinstance(pred_row, pl.DataFrame):
            pred_row = pred_row.to_pandas()
        if not isinstance(pred_row, pd.DataFrame):
            raise ValueError("predict_row_fn must return pandas or polars DataFrame")

        y_pred_flat = pred_row.to_numpy(dtype=float).reshape(-1)

        # 길이가 서로 다르면 스코어 불가 → 건너뜀(또는 에러)
        if y_true_flat.size == 0 or y_true_flat.size != y_pred_flat.size:
            continue

        # 전체 MSE
        mses.append(masked_mse(y_true_flat, y_pred_flat))

        # lag별 MSE (슬라이스)
        offs = 0
        for k, b in zip((1,2,3,4), (b1,b2,b3,b4)):
            keep = [c for c in b.columns if c not in ("date_id","label_date_id")]
            n = len(keep)
            if n == 0: 
                continue
            yt = y_true_flat[offs:offs+n]
            yp = y_pred_flat[offs:offs+n]
            mses_by_lag[k].append(masked_mse(yt, yp))
            by_lag_arrays[k]["y_true"].append(yt[np.isfinite(yt) & np.isfinite(yp)])
            by_lag_arrays[k]["y_pred"].append(yp[np.isfinite(yt) & np.isfinite(yp)])
            offs += n

    overall = float(np.nanmean(mses)) if mses else np.nan
    per_lag = {k: (float(np.nanmean(v)) if v else np.nan) for k,v in mses_by_lag.items()}

    if return_arrays:
        # concatenate for plotting
        for k in by_lag_arrays:
            by_lag_arrays[k]["y_true"] = (np.concatenate(by_lag_arrays[k]["y_true"])
                                          if by_lag_arrays[k]["y_true"] else np.array([]))
            by_lag_arrays[k]["y_pred"] = (np.concatenate(by_lag_arrays[k]["y_pred"])
                                          if by_lag_arrays[k]["y_pred"] else np.array([]))
        return {"overall": overall, "per_lag": per_lag, "by_lag_arrays": by_lag_arrays}

    return {"overall": overall, "per_lag": per_lag}

def evaluate_preds_by_lag(
    test_csv_path: str,
    lag_dir: str,
    predict_row_fn
):
    out = evaluate_mse_over_test2(
        test_csv_path=test_csv_path,
        lag_dir=lag_dir,
        predict_row_fn=predict_row_fn,
        return_arrays=True
    )
    return out["by_lag_arrays"]


@dataclass 
class MSEReport:
    overall_mse: float 
    per_lag_mse: Dict
    per_target_mse: Optional=None 

#########################
# Hypermarameter search  
#########################
@dataclass
class ModelSpec:
    name: str
    build_fn: Callable # returns model object 
    train_fn: Callable 
    predict_fn: Callable 
    default_space: Dict 

def register_model(spec:ModelSpec):
    MODEL_REGISTRY[spec.name] = spec 



#########################
# Plot 
#########################
def plot_true_vs_pred(results, title="True vs Predicted", max_points_to_plot=5000):
    """
    Scatter plot to compare true values and predictions.
    
    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground truth values.
    y_pred : array-like, shape (n_samples,)
        Predicted values from the model.
    title : str, optional
        Title for the plot.
    max_points_to_plot : int, optional
        Number of points to plot (random sample for readability if data is too large).
    """
    colors = {1: "blue", 2: "green", 3: "orange", 4: "red"}

    plt.figure(figsize=(7,7))
    for lag, data in results.items():
        y_true, y_pred = data["y_true"], data["y_pred"]

        if len(y_true) > max_points_to_plot:
            idx = np.random.choice(len(y_true), max_points_to_plot, replace=False)
            y_true, y_pred = y_true[idx], y_pred[idx]

        plt.scatter(y_true, y_pred, alpha=0.3, s=10, c=colors[lag], label=f"Lag {lag}")

    # y=x 기준선
    all_true = np.concatenate([results[l]["y_true"] for l in results])
    plt.plot([all_true.min(), all_true.max()],
             [all_true.min(), all_true.max()],
             'k--', lw=2, label="ideal y=x")

    plt.xlabel("True values")
    plt.ylabel("Predicted values")
    plt.title("True vs Predicted by Lag")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_true_vs_pred_by_lag_arrays(by_lag_arrays, sample_size=3000):
    colors = {1:"C0", 2:"C2", 3:"C1", 4:"C3"}
    plt.figure(figsize=(7,7))
    mins, maxs = [], []
    for lag in (1,2,3,4):
        yt = by_lag_arrays[lag]["y_true"]
        yp = by_lag_arrays[lag]["y_pred"]
        if len(yt)==0: 
            continue
        if len(yt)>sample_size:
            idx = np.random.choice(len(yt), sample_size, replace=False)
            yt, yp = yt[idx], yp[idx]
        plt.scatter(yt, yp, s=10, alpha=0.3, label=f"Lag {lag}", c=colors[lag])
        mins.append(np.nanmin(yt)); mins.append(np.nanmin(yp))
        maxs.append(np.nanmax(yt)); maxs.append(np.nanmax(yp))
    if mins and maxs:
        lo, hi = min(mins), max(maxs)
        plt.plot([lo,hi],[lo,hi],'k--',lw=2,label='y=x')
    plt.xlabel("True"); plt.ylabel("Prediction")
    plt.title("True vs Predicted by Lag")
    plt.legend(); plt.grid(True); plt.show()
