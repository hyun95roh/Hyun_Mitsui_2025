import torch
import torch.nn as nn 
import numpy as np 
import time 
import pandas as pd 
from typing import Tuple, List, Dict, Callable, Optional, Protocol, Iterable  
from dataclasses import dataclass
from models import LSTMForecaster, CNNLSTMForecaster, MinMaxScalerTS, StandardScaler2D
from models import WindowDataset, tune_weight_global, tune_weight_per_target, weighted_ensemble
from models import lstm_predict, arima_predict
from collections import defaultdict

@dataclass
class TrainConfig:
    input_len: int = 64
    output_len: int = 1
    batch_size: int = 64
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-3
    channels: int = 64
    blocks: int = 3 
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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

def masked_mse_per_target(pred, target):
    mask = torch.isfinite(target)
    target = torch.nan_to_num(target)
    pred   = torch.nan_to_num(pred)
    sq = (pred - target)**2 * mask
    denom = mask.sum().clamp_min(1)
    return sq.sum() / denom

def make_dataloader(X, Y, bs):
    ds = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32),
                                        torch.tensor(Y, dtype=torch.float32))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=False)

def build_scheduler(optimizer, policy, steps_per_epoch, epochs, base_lr):
    if policy=="clr":  # triangular CLR
        # simple triangular cycle over one epoch
        def lr_lambda(step):
            t = (step % steps_per_epoch) / max(1, steps_per_epoch-1)
            return 0.1 + 0.9*(1 - abs(2*t-1))  # between 0.1x and 1.0x
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if policy=="onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr,
                                                   steps_per_epoch=steps_per_epoch,
                                                   epochs=epochs, pct_start=0.3)
    return None

def train_one_model_masked(model, x_scaler, cfg, Xtr, Ytr, Xva, Yva, lr_policy="constant"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=getattr(cfg, "weight_decay", 0.0))
    train_loader = make_dataloader(Xtr, Ytr, cfg.batch_size)
    val_loader   = make_dataloader(Xva, Yva, cfg.batch_size)
    sched = build_scheduler(opt, lr_policy, steps_per_epoch=len(train_loader), epochs=cfg.epochs, base_lr=cfg.lr)

    best = np.inf; bad=0
    for ep in range(cfg.epochs):
        # train
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)[:,0,:]
            loss = masked_mse_per_target(out, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched: sched.step()
        # val
        model.eval()
        with torch.no_grad():
            v_losses=[]
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)[:,0,:]
                v_losses.append(masked_mse_per_target(out, yb).item())
            v = float(np.mean(v_losses))
        model.train()

        if v < best - 1e-7:
            best, bad = v, 0
            best_state = {k:v.clone() if torch.is_tensor(v) else v for k,v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= getattr(cfg, "patience", 10): break
    model.load_state_dict(best_state)
    return best


def train_eval_fn(build_model_fn, cfg, X_hist, Y_hist, trial=None):
    # 1) make windows from X_hist/Y_hist with cfg.input_len (do NOT drop rows; use masked loss)
    Xw, Yw = make_windows(Y_hist, cfg.input_len, X=X_hist)  # your existing helper
    # 2) scale X
    N,T,F = Xw.shape
    model, x_scaler = build_model_fn(cfg, num_features=F, num_targets=Yw.shape[1])
    X2d = Xw.reshape(-1,F)
    X2d = x_scaler.fit_transform(X2d)
    X2d = np.nan_to_num(X2d)
    Xw = X2d.reshape(N,T,F)

    # 3) split
    n_val = max(1, int(0.15*N))
    Xtr, Ytr = Xw[:-n_val], Yw[:-n_val]
    Xva, Yva = Xw[-n_val:], Yw[-n_val:]

    # 4) train with masked loss, lr policy = constant/CLR/onecycle (implement small helper)
    val_mse = train_one_model_masked(model, x_scaler, cfg, Xtr, Ytr, Xva, Yva, lr_policy=cfg.lr_policy)
    return val_mse


#########################
# Particular Trainer 
#########################
def train_lstm(
        Y: np.ndarray,  #(N,) or (N, Target)
        cfg: TrainConfig, 
        X: Optional[np.ndarray] = None, # (N, F) or None (univariate AR)
        val_share_min: int = 128, 
        scale_X: bool = True, 
) -> Tuple[LSTMForecaster, Optional[StandardScaler2D], Optional[MinMaxScalerTS], dict]:
    """
    Trains ONE LSTM in either mode based on shapes:
    - Univariate AutoRegressive: Y.ndim==1 and X is None -> input_size=1, num_targets=1
    - Multivariate: X provided (N,F) -> input_size=F, num_targets= Y.shape[1]
    - Multi-output univeriate is also supported if Y.ndim==2 and X is None.
    Returns: (model_on_cpu, X_scaler or None, Y_scaler or None, meta)
    """
    print(f"Your device: {cfg.device}")
    Y = np.asarray(Y, dtype= np.float32)
    if Y.ndim == 1:
        Y = Y[:, None]  # (N, 1)
    N, TGT = Y.shape 

    # Choose inputs 
    if X is None:
        X = Y  # univariate Autoregressive
    else: 
        X = np.asarray(X, dtype= np.float32) 
        assert len(X) == N, "X and Y must have same length"
    
    # Optionally scale features col-wise; leave Y as returns
    X_scaler = None 
    if scale_X:
        finite_rows = np.isfinite(X).all(axis=1) 
        if finite_rows.sum() < 10:
            raise ValueError("Not enough rows to fit X scaler") 
        X_scaler = StandardScaler2D().fit(X[finite_rows]) 
        Xs = X_scaler.transform(X) 
    else:
        Xs = X 
    
    # Trim to rows where we can form windows 
    finite_rows = np.isfinite(Xs).all(axis=1) & np.isfinite(Y).all(axis=1) 
    if finite_rows.sum() < cfg.input_len + cfg.output_len + 10:
        raise ValueError("Not enough finite joint rows to train") 
    Xs, Y = Xs[finite_rows], Y[finite_rows] 

    # Split train/val 
    val_len = max(val_share_min, cfg.input_len + cfg.output_len)
    val_len = min(val_len, max(1, len(Xs)//5)) 
    X_tr, X_val = Xs[:-val_len], Xs[-val_len:]
    Y_tr, Y_val = Y[:-val_len], Y[-val_len:]

    tr_ds = WindowDataset(X_tr, Y_tr, cfg.input_len, cfg.output_len)
    val_ds = WindowDataset(X_val, Y_val, cfg.input_len, cfg.output_len) 
    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # Model setting
    input_size = Xs.shape[1] 
    num_targets = TGT     

    model = LSTMForecaster(
        input_size = input_size, 
        hidden_size = cfg.hidden_size, 
        num_layers = cfg.num_layers, 
        dropout = cfg.dropout, 
        output_len = cfg.output_len, 
        num_targets = num_targets 
    ).to(cfg.device) 

    # Optimizer and Loss function setting
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss() 

    # Training Epoches:
    best, bad, best_state = np.inf, 0, None 
    for _ in range(cfg.epochs): 
        # train 
        model.train() 
        for xb, yb in tr_dl:
            xb = xb.to(cfg.device)  # (B, T, F)
            yb = yb.to(cfg.device).squeeze(1)  # (B, TGT) since output_len=1 typical.
            opt.zero_grad()
            pred = model(xb).squeeze(1) 
            loss = loss_fn(pred, yb) 
            loss.backward()
            opt.step() 
        
        # Validate
        model.eval() 
        val_loss = 0.0 
        n = 0 
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(cfg.device)
                yb = yb.to(cfg.device).squeeze(1) 
                pred = model(xb).squeeze(1)
                val_loss += loss_fn(pred, yb).item() * len(xb)
                n += len(xb) 
            val_loss /= max(1, n)
            
        if val_loss + 1e-12 < best:
            best, bad = val_loss, 0 
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience: break 
    
    print(f" >> val_loss(best): {best}") 
    model.load_state_dict(best_state)
    model.cpu()

    meta = dict(input_size = input_size, num_targets=num_targets) 
    # Y_scaler is None (not used); returned for symmetry
    return model, X_scaler, None, meta 


def train_cnnlstm(
        Y: np.ndarray, 
        cfg, 
        X: Optional=None, 
        model_kwargs: Optional = None, 
        device: str = 'cpu', 
        lr: float=1e-3, 
        batch_size: int = 128, 
        val_frac: float = 0.1, 
        seed: int = 42, 
        verbose: bool = True, 
        use_tqdm: bool = True, 
        patience: int = 10,            # Early stopping 
        clip_grad_norm: float = 1.0,   # gradient clipping 
        scheduler_patience: int = 5,   # LR scheduler patience 
        scheduler_factor: float = 0.5, # LR scheduler factor 
        partially_finite_target: bool = False # If True, allow windows where some target values are NaN
) -> Tuple[nn.Module, object, dict]:
    """
    Returns: (model, x_scaler, meta)
    - x_scaler: reuse your StandardScaler2D or similar (pass-through OK if already scaled)
    - meta: dict with training stats. 
    """
    print(f"Your device: {cfg.device}")
    import warnings
    from sklearn.preprocessing import StandardScaler
    try:
        from tqdm.auto import tqdm
    except Exception:
        use_tqdm = False
    torch.manual_seed(seed) 
    np.random.seed(seed) 

    # Instead of dropping windows, we can mask the loss where target is NaN
    # This is useful if you have many targets and some are missing at random.
    # Note: if all targets are NaN, that window is still dropped.
    def masked_mse(pred, target):
        # pred/target: (B, Targets)
        mask = torch.isfinite(target)
        # replace NaNs in target (won't be used where mask=0)
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        pred   = torch.nan_to_num(pred,   nan=0.0, posinf=0.0, neginf=0.0)
        sq = (pred - target)**2
        sq = sq * mask
        denom = mask.sum().clamp_min(1) # clamp_min(1) to avoid div0.
        return sq.sum() / denom
    
    # Window  
    Xw, Yw = make_windows(Y, input_len=cfg.input_len, X=X) 
    N, T, F = Xw.shape 
    num_targets = Yw.shape[1] 

    # ---- scaling (flatten to 2D for sklearn)
    x_scaler = getattr(cfg, "x_scaler", None)
    if x_scaler is None:
        x_scaler = StandardScaler()
    X2d = Xw.reshape(-1, F)
    X2d = x_scaler.fit_transform(X2d)
    # guard against inf/nan from scaler
    X2d = np.nan_to_num(X2d, nan=0.0, posinf=0.0, neginf=0.0)
    Xw = X2d.reshape(N, T, F)

    # Split train/val 
    n_val = max(1, int(N*val_frac)) 
    Xtr, Ytr = Xw[:-n_val], Yw[:-n_val] 
    Xva, Yva = Xw[-n_val:], Yw[-n_val:]

    # Build model 
    model_kwargs = dict(model_kwargs or {})
    model_kwargs.setdefault("input_size", F)
    model_kwargs.setdefault("num_targets", num_targets)
    model_kwargs.setdefault("conv_channels", 128)
    model_kwargs.setdefault("kernel_size", 5)
    model_kwargs.setdefault("num_conv_layers", 2)
    model_kwargs.setdefault("lstm_hidden", cfg.hidden_size)
    model_kwargs.setdefault("lstm_layers", getattr(cfg, "num_layers", 1))
    model_kwargs.setdefault("dropout", cfg.dropout)
    model_kwargs.setdefault("output_len", cfg.output_len)

    model = CNNLSTMForecaster(**model_kwargs).to(device) 

    # Train 
    ds_tr = TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)) 
    ds_va = TensorDataset(torch.tensor(Xva), torch.tensor(Yva)) 
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True) 
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False) 

    opt = torch.optim.AdamW(model.parameters(), lr=lr) 
    loss_fn = torch.nn.MSELoss() 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=scheduler_factor, patience=scheduler_patience, verbose=verbose
    )
    best_val = float('inf') 
    best_state =None 
    bad_epochs = 0 
    history = defaultdict(list) 

    epoch_iter = range(cfg.epochs) 
    if use_tqdm:
        epoch_iter = tqdm(epoch_iter, desc='Training CNN+LSTM', unit='epoch')

    for epoch in epoch_iter:
        #----- Train 
        model.train()
        tr_loss_sum = 0.0 
        for xb, yb in dl_tr:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            out = model(xb)[:, 0, :]                  # (B, Targets)
            loss = masked_mse(out, yb) if partially_finite_target==True else loss_fn(out, yb)
            if torch.isnan(loss):
                warnings.warn("NaN loss encountered; reducing LR and continuing.")
                for g in opt.param_groups:
                    g["lr"] *= 0.5
            loss.backward()
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            opt.step()
            tr_loss_sum += loss.item() * len(xb)
        
        tr_loss = tr_loss_sum / len(ds_tr) 

        #----- Validate 
        model.eval()
        va_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device); yb = yb.to(device)
                out = model(xb)[:, 0, :]
                loss = loss_fn(out, yb)
                va_loss_sum += loss.item() * len(xb)
        va_loss = va_loss_sum / len(ds_va)

        #----- Book-keeping 
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["lr"].append(opt.param_groups[0]["lr"])

        if verbose and not use_tqdm:
            print(f"epoch {epoch+1}/{cfg.epochs} | train {tr_loss:.6f} | val {va_loss:.6f} | lr {opt.param_groups[0]['lr']:.2e}")
        if use_tqdm:
            epoch_iter.set_postfix(train=tr_loss, val=va_loss, lr=opt.param_groups[0]['lr'])


        #----- Early stopping 
        scheduler.step(va_loss) 
        if va_loss < best_val - 1e-8:
            best_val = va_loss 
            best_state = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            bad_epochs = 0 
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"Early stopping: {bad_epochs} bad epochs")
                break
    if best_state is not None:
        model.load_state_dict(best_state) 

    print(f"Best val_loss in training: {best_val}")
    meta = dict(best_val=best_val, F=F, T=T) 
    model.eval() 

    return model, x_scaler, meta 



########################
# Ensemble
########################
def run_ensemble(
        targets_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        cfg: TrainConfig,
        feature_cols = None,
        use_multivariate: bool = False,
        arima_jobs: int = 1,
        weight_mode: str = "global",
        tune_window: int = 200,
        max_lag: int = 4):
    """
    - Multivariate branch trains ONE LSTM over all targets, tunes ensemble weights
      on a validation window that EXCLUDES the label NaN tail (max_lag + 1).
    - Univariate branch trains per-target LSTM and blends with per-target ARIMA.
    """

    # ---------- MULTIVARIATE BRANCH ----------
    if use_multivariate:
        # 1) Pick features
        feature_cols = feature_cols or [c for c in prices_df.columns if c != "date_id"]

        # 2) Align features to target index (expects prices_df to have a date_id column)
        X = prices_df.set_index("date_id").loc[targets_df.index, feature_cols].to_numpy(float)
        Y = targets_df.to_numpy(float)   # (N, TGT)

        # 3) Train one multivariate LSTM
        model, xscaler, _, meta = train_lstm(Y, cfg, X=X)

        # 4) Build a tuning window that EXCLUDES tail rows where targets are NaN by definition
        tune_end = len(Y) - (max_lag + 1)
        if tune_end <= cfg.input_len + 5:
            raise ValueError("Not enough non-NaN history to tune. Reduce input_len or add more data.")
        tune_start = max(0, tune_end - tune_window)
        idx_tune = slice(tune_start, tune_end)
        X_tune, Y_tune = X[idx_tune], Y[idx_tune]

        # 5) Rolling 1-step predictions inside the tuning window
        print("Initiate rolling 1-step predictions...")
        yl_list, ya_list, yt_list = [], [], []
        for t in range(cfg.input_len, len(Y_tune)):
            print(f" - Loop: {t}/{len(Y_tune)-1}") 
            X_hist = X_tune[:t]
            Y_hist = Y_tune[:t]

            # LSTM 1-step
            yhat_l = lstm_predict(model, xscaler, Y_hist=Y_hist, input_len=cfg.input_len, X_hist=X_hist)  # (TGT,)

            # ARIMA 1-step (rename: fit_predict_arima_block)
            yhat_a = arima_predict(pd.DataFrame(Y_hist, columns=targets_df.columns),
                                             horizon=1, n_jobs=arima_jobs)                               # (TGT,)

            # True next step
            yt = Y_tune[t]  # (TGT,)

            yl_list.append(yhat_l); ya_list.append(yhat_a); yt_list.append(yt)

        # 6) Stack & filter finite rows
        YL = np.vstack(yl_list)  # (W, TGT)
        YA = np.vstack(ya_list)  # (W, TGT)
        YT = np.vstack(yt_list)  # (W, TGT)

        finite_rows = np.isfinite(YT).all(axis=1) & np.isfinite(YL).all(axis=1) & np.isfinite(YA).all(axis=1)
        YTf, YLf, YAf = YT[finite_rows], YL[finite_rows], YA[finite_rows]

        # 7) If nothing remains, fallback to a safe weight; else tune
        if len(YTf) == 0:
            # Prefer LSTM if at least LSTM+truth rows were finite; else default 0.5
            finite_rows_lstm = np.isfinite(YT).all(axis=1) & np.isfinite(YL).all(axis=1)
            w = 1.0 if finite_rows_lstm.any() else 0.5
        else:
            if weight_mode == "per_target":
                w = tune_weight_per_target(YTf, YLf, YAf)      # (TGT,)
            else:
                w = tune_weight_global(YTf, YLf, YAf)          # scalar

        # 8) Final next-step predictions at the end of full Y/X
        yhat_next_l = lstm_predict(model, xscaler, Y_hist=Y, input_len=cfg.input_len, X_hist=X)          # (TGT,)
        yhat_next_a = arima_predict(targets_df, horizon=1, n_jobs=arima_jobs)                  # (TGT,)
        yhat_next   = weighted_ensemble(yhat_next_l, yhat_next_a, w)                                     # (TGT,)

        return dict(pred_lstm=yhat_next_l, pred_arima=yhat_next_a, pred_ens=yhat_next, w=w)

    # ---------- UNIVARIATE BRANCH ----------
    else:
        preds_l, preds_a, preds_e, ws = [], [], [], []
        for col in targets_df.columns:
            y = targets_df[col].to_numpy(float)

            # LSTM (univariate AR): X=None
            model, xscaler, _, meta = train_lstm(y, cfg, X=None)
            yhat_l = lstm_predict(model, xscaler, Y_hist=y, input_len=cfg.input_len)  # (1,)

            # ARIMA (rename: fit_predict_arima_block)
            yhat_a = arima_predict(targets_df[[col]], horizon=1, n_jobs=1)[0]

            # Simple default weight; replace with per-target tuning if desired
            w = 0.5

            preds_l.append(yhat_l[0])
            preds_a.append(yhat_a)
            ws.append(w)
            preds_e.append(weighted_ensemble(yhat_l[0], yhat_a, w))

        return dict(
            pred_lstm=np.array(preds_l, dtype=float),
            pred_arima=np.array(preds_a, dtype=float),
            pred_ens=np.array(preds_e, dtype=float),
            w=np.array(ws, dtype=float),
        )