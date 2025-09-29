# Hyperparameter optimization

import numpy as np 
import optuna 
import time 
import copy
from dataclasses import dataclass 

#===================================

@dataclass
class HPOConfig:
    max_trials: int=30
    timeout: int=0                   # 0 = no timeout 
    pruner: str='median'             # or 'none'
    direction: str = 'minimize'      # MSE 
    seed: int = 42 

def suggest_space(trial, base_cfg):
    cfg = copy.deepcopy(base_cfg) 
    # Common Knobs across LSTM/CNN-LSTM/TCN: 
    cfg.input_len = trial.suggest_int('input_len', 32, 256, step=32)
    cfg.dropout = trial.suggest_float('dropout', 0.0, 0.5) 
    cfg.lr = trial.suggest_float('lr', 1e-5, 3e-3, log=True)
    cfg.weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-3, log=True)
    cfg.batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    if hasattr(cfg, 'hidden_size'):
        cfg.hidden_size = trial.suggest_categorical('hidden_size', [128,256, 384, 512])
    
    # CNN front-end knobs (if model uses them; ignored otherwise)
    cfg.cnn_kernel = trial.suggest_categorical("cnn_kernel", [3,5,7])
    cfg.cnn_dilation = trial.suggest_categorical('cnn_dilation', [1,2,4,8])

    # TCN-specific (if used)
    cfg.tcn_blocks = trial.suggest_int('tcn_blocks', 2, 6)
    cfg.tcn_channels = trial.suggest_categorical('tcn_channels', [128, 256, 384])
    cfg.tcn_kernel = trial.suggest_categorical('tcn_kernel', [2,3,5])

    return cfg 

def objective_factory(build_model_fn, train_eval_fn, base_cfg, X_hist, Y_hist):
    """ 
    build_model_fn(cfg) -> model, scaler 
    train_eval_fn(model, scaler, cfg, X_hist, Y_hist) -> float(val_MSE)
    """

    def objective(trial):
        cfg = suggest_space(trial, base_cfg)
        # Optional schedule policy
        cfg.lr_policy = trial.suggest_categorical('lr_policy', ['constant','clr','onecycle'])

        # Fit + Val score 
        val_mse = train_eval_fn(build_model_fn, cfg, X_hist, Y_hist, trial=trial)
    return objective 

def run_optuna(build_model_fn, train_eval_fn, base_cfg, X_hist, Y_hist, hpo_cfg=HPOConfig()):
    pruner = optuna.pruners.MedianPruner() if hpo_cfg.pruner=="median" else optuna.pruners.NopPruner()
    study = optuna.create_study(direction=hpo_cfg.direction, pruner=pruner, sampler=optuna.samplers.TPESampler(seed=hpo_cfg.seed))
    study.optimize(objective_factory(build_model_fn, train_eval_fn, base_cfg, X_hist, Y_hist),
                   n_trials=hpo_cfg.max_trials,
                   timeout=hpo_cfg.timeout if hpo_cfg.timeout>0 else None)
    return study