import os, sys
import logging, traceback 
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from datetime import datetime
import gc
import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from dataprep import WindowDataset, collate_informer
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction as TSTP


# Import your modules
from dataprep import dataprep, build_informer_datasets, collate_informer
from configs import TrainConfig
from models import LSTMForecaster, CNNLSTMForecaster, TCNForecaster, FEDForecaster, InForecaster
from models import build_derits
from trainer import LSTMTrainer, CNNLSTMTrainer, FEDTrainer, TCNTrainer, TSTPTrainer, DERTrainer
from trainer import TimesFMTrainer, TimesFMTrainerConfig, InTrainer
from spectral import (apply_freq_derivative, inverse_freq_derivative,
                      ComplexDepthwiseConvFreq, ComplexLinear, BandMask)

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  #logging.INFO/.DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Map string class names to actual classes
MODEL_CLASSES = {
    'LSTM': LSTMForecaster,
    'CNNLSTM': CNNLSTMForecaster,
    'TCN': TCNForecaster,
    'FED': FEDForecaster,
    'TSTP': TSTP,
    'INF': InForecaster, 
}

TRAINER_CLASSES = {
    'LSTMTrainer': LSTMTrainer,
    'CNNLSTMTrainer': CNNLSTMTrainer,
    'FEDTrainer': FEDTrainer,
    'TCNTrainer': TCNTrainer,
    'TSTPTrainer': TSTPTrainer,
    'TimesFMTrainer': TimesFMTrainer,
    'InTrainer': InTrainer,
}

# Function to set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Set random seed to {seed} for reproducibility.")

# Function to clean memory
def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    logging.info("Performed memory cleaning.")

# Function to monitor memory
def monitor_memory():
    if torch.cuda.is_available():
        mem_info = torch.cuda.mem_get_info()
        used_mem = (mem_info[1] - mem_info[0]) / (1024 ** 3)  # GB
        total_mem = mem_info[1] / (1024 ** 3)
        logging.info(f"GPU Memory: {used_mem:.2f} GB used out of {total_mem:.2f} GB")
        return used_mem, total_mem
    else:
        import resource
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)  # MB
        logging.info(f"CPU Memory Usage: {mem_usage:.2f} MB")
        return mem_usage, None

# Optuna objective function 

# ---------------- TimesFM-specific objective helper ----------------
def objective_timesfm(trial, cfg, X_raw, Y_raw, checkpoint_dir):
    """
    Use TimesFMTrainer for a *representative* target series to score a trial.
    This keeps Optuna fast and still gives useful signal for hyperparameters.
    If you want to sweep on multiple targets, loop here and average their metrics.
    """

    # Map from hydra config to TimesFMTrainerConfig
    params = dict(cfg.model.params) if hasattr(cfg.model, "params") else {}
    tfm_cfg = TimesFMTrainerConfig(
        model_name      = params.get("model_name", "google/timesfm-1.0-200m-pytorch"),
        context_length  = int(params.get("context_length", params.get("input_len", 512))),
        horizon_length  = int(params.get("horizon_length", params.get("output_len", 5))),
        freq_type       = int(params.get("freq_type", 0)),
        batch_size      = int(params.get("batch_size", 32)),
        epochs          = int(params.get("epochs", 20)),
        lr              = float(params.get("lr", getattr(cfg.model, "lr", 1e-4))),
        weight_decay    = float(params.get("weight_decay", 0.0)),
        use_quantile_loss = bool(params.get("use_quantile_loss", False)),
        out_dir         = os.path.join(checkpoint_dir, "timesfm"),
        single_step_pred= bool(params.get("single_step_pred", True)),
    )

    # Simple trial sampling knobs for Optuna (optional)
    tfm_cfg.lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    tfm_cfg.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Choose one representative target to evaluate (can be changed via cfg.experiment.target_index)
    target_idx = 0
    series = Y_raw[:, target_idx].astype(np.float32)  # univariate

    trainer = TimesFMTrainer(tfm_cfg)
    result = trainer.train_one_series(series, target_name=f"target_{target_idx}")

    # Extract a scalar metric for Optuna
    m = None
    if isinstance(result, dict):
        m = result.get("best_val") or result.get("val_loss")
    if m is None:
        # last-resort: very large penalty if nothing returned
        return 1e9
    return float(m)


def objective_informer(trial, cfg, X_raw, Y_raw, checkpoint_dir):
    params = {
        'input_size': X_raw.shape[1],
        'output_size': Y_raw.shape[1],
        'output_len': int(cfg.model.params.get('output_len', 5)),
        'input_len': trial.suggest_int('input_len', 32, 128, step=16),
        'label_len': trial.suggest_int('label_len', 16, 64, step=16),
        'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 384, 512]),
        'num_layers': trial.suggest_int('e_layers', 2, 4),
        'd_layers': trial.suggest_int('d_layers', 1, 3),
        'nhead': trial.suggest_categorical('n_heads', [4, 8]),
        'dim_feedforward': trial.suggest_int('d_ff', 512, 2048, step=256),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3, step=0.05),
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
        'attn': 'prob',
        'distil': True,
        #'mix': True,
    }
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'AdamW'])
    lr_policy = trial.suggest_categorical('lr_policy', ['constant', 'clr', 'onecycle'])

    # Validate hidden_size
    if params['hidden_size'] % params['nhead'] != 0:
        logging.warning(f"Trial {trial.number}: Invalid hidden_size={params['hidden_size']} for n_heads={params['nhead']}. Pruning.")
        raise optuna.TrialPruned()

    cfg.model.params.update(params)
    cfg.model.lr = lr
    cfg.model.optimizer = optimizer
    cfg.model.lr_policy = lr_policy

    logging.info(f"Trial {trial.number} started with params: {params}, lr={lr}, optimizer={optimizer}, lr_policy={lr_policy}")

    clean_memory()
    pre_mem = monitor_memory()

    try:
        x_nans = np.isnan(X_raw).sum()
        y_nans = np.isnan(Y_raw).sum()
        logging.info(f"Trial {trial.number}: X_raw shape={X_raw.shape}, Y_raw shape={Y_raw.shape}, NaNs X={x_nans}, Y={y_nans}")
        if len(X_raw) < params['input_len'] + params['output_len']:
            logging.warning(f"Trial {trial.number}: Insufficient data length {len(X_raw)} for input_len={params['input_len']} + output_len={params['output_len']}")
            raise optuna.TrialPruned()

        #train_loader, val_loader, _ = build_informer_datasets(
        #    TrainConfig(**params, lr=lr, optimizer=optimizer, model_type='InForecaster', optuna_trial=trial),
        #    X_raw,
        #    Y_raw,
        #    use_time_feats=True
        #)

        trainer_class = TRAINER_CLASSES['InTrainer']
        train_cfg = TrainConfig(
            **params,
            lr=lr,
            optimizer=optimizer,
            model_type='InForecaster',
            optuna_trial=trial
        )
        
        #logging.info(f"Trial {trial.number}: Creating InTrainer with device={train_cfg.device}")
        #trainer = trainer_class(train_cfg)
        trainer = InTrainer(train_cfg, criterion=nn.MSELoss())
        #val_loss = trainer.train(train_loader=train_loader, val_loader=val_loader)
        
        val_loss = trainer.train(X=X_raw, Y=Y_raw, train_loader=None, val_loader=None)

        trainer.save_best_model(
            os.path.join(
                checkpoint_dir,
                f"INF_trial{trial.number}_trial{trial.number}_{val_loss:.6f}.pt"))

        return val_loss

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"Trial {trial.number} failed: GPU Out of Memory Error: {str(e)}\n{traceback.format_exc()}")
        if params['batch_size'] > 16:
            trial.set_user_attr('retry_batch_size', params['batch_size'] // 2)
            trial.set_user_attr('retry_hidden_size', params.get('hidden_size', 256) // 2)
            logging.info(f"Retrying with batch_size={params['batch_size'] // 2}, hidden_size={params.get('hidden_size', 256) // 2}")
            raise optuna.TrialPruned()
        raise
    except Exception as e:
        logging.error(f"Trial {trial.number} failed: {str(e)}\n{traceback.format_exc()}")
        raise optuna.TrialPruned()
    finally:
        try:
            from accelerate.state import AcceleratorState
            AcceleratorState._reset_state()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
        gc.collect()
        post_mem = monitor_memory()
        clean_memory()


def object_derits(trial, cfg, X_raw, Y_raw, checkpoint_dir):
    # Here cfg is model-specific *.yaml files. 
    """
    Optuna objective for DERITS. Mirrors the structure of objective_informer/objective_timesfm:
    - makes windows
    - splits train/val
    - samples a few hyperparams from trial
    - trains with DERTrainer
    - reports/prunes on val loss
    - saves best checkpoint under checkpoint_dir
    """
    gc.collect() 
    clean_memory() # clean memory 
    
    # ---------- resolve sizes from cfg / params ----------
    p = cfg.model.params
    search_space = cfg.model.search_space  # Access search space from config
    orders_map = {"012": (0,1,2), "01": (0,1), "12": (1,2), "02": (0,2)}
    batch_size  = int(p.get('batch_size', 32))
    output_len = int(p.get('output_len', 5))
    input_size = int(p.get('input_size',  X_raw.shape[1]))
    output_size= int(p.get('output_size', Y_raw.shape[1]))

    # ---------- hyperparams via trial using search_space ----------
    input_len = trial.suggest_int('input_len', search_space.input_len.low, search_space.input_len.high, step=search_space.input_len.step) 
    orders = orders_map[trial.suggest_categorical('orders', search_space.orders.choices)]
    depth = trial.suggest_int('depth', search_space.depth.low, search_space.depth.high, step=search_space.depth.step)
    kernel_size = trial.suggest_categorical('kernel_size', search_space.kernel_size.choices)
    dropout = trial.suggest_float('dropout', search_space.dropout.low, search_space.dropout.high, step=search_space.dropout.step)
    rnn_mul = trial.suggest_categorical('rnn_hidden_mul', search_space.rnn_hidden_mul.choices)

    lr          = float(getattr(cfg, 'lr', 2e-4))
    wd          = float(getattr(cfg, 'wd', 1e-4))
    num_workers = int(getattr(cfg, 'num_workers', 4))
    amp         = bool(getattr(cfg, 'amp', True))

    # Optional DERITS regularizers
    band_l1     = trial.suggest_float('band_l1', search_space.band_l1.low, search_space.band_l1.high)#float(getattr(cfg, 'band_l1', 1e-4))
    hf_fraction = trial.suggest_float('hf_fraction', search_space.hf_fraction.low, search_space.hf_fraction.high)#float(getattr(cfg, 'hf_fraction', 0.2))
    hf_weight   = trial.suggest_float('hf_weight', search_space.hf_weight.low, search_space.hf_weight.high, step=0.05)#float(getattr(cfg, 'hf_weight', 1e-4))
    grad_clip   = trial.suggest_float('grad_clip', search_space.grad_clip.low, search_space.grad_clip.high, step=0.5)#float(getattr(cfg, 'grad_clip', 0.0))

    # ---------- windows ----------
    ds = WindowDataset(X_full=X_raw, Y_full=Y_raw, input_len=input_len, output_len=output_len)
    Xw, Yw = ds.x, ds.y  # numpy float32: [N, T, F], [N, H, D]

    # guard: ensure shapes consistent with model expectations
    assert Xw.shape[2] == input_size,  f"input_size mismatch: {Xw.shape[2]} vs {input_size}"
    assert Yw.shape[1] == output_len,  f"output_len mismatch: {Yw.shape[1]} vs {output_len}"
    assert Yw.shape[2] == output_size, f"output_size mismatch: {Yw.shape[2]} vs {output_size}"

    # ---------- split train/val like your other objectives ----------
    val_ratio = float(getattr(cfg, 'val_ratio', 0.2))
    n = len(Xw); n_val = max(1, int(n * val_ratio))
    n_tr = n - n_val
    Xtr, Ytr = Xw[:n_tr], Yw[:n_tr]
    Xva, Yva = Xw[n_tr:], Yw[n_tr:]

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    val_ds   = TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva))

    # ---------- build per-run cfg for the trainer ----------
    class _DERCfg:
        pass
    _c = _DERCfg()
    _c.L = input_len; _c.H = output_len; _c.C_in = input_size; _c.D = output_size
    _c.orders = orders; _c.depth = depth; _c.kernel_size = kernel_size; _c.dropout = dropout
    _c.batch_size = batch_size; _c.num_workers = num_workers
    _c.lr = lr; _c.wd = wd; _c.amp = amp
    _c.band_l1 = band_l1; _c.hf_fraction = hf_fraction; _c.hf_weight = hf_weight; _c.grad_clip = grad_clip
    _c.checkpoint_path = os.path.join(checkpoint_dir, "derits_best.pt")
    _c.rnn_hidden_mul = rnn_mul
    _c.epochs = int(getattr(cfg, 'epochs', 20))  # for parity with other objectives
    _c.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- trainer ----------
    trainer = DERTrainer(cfg=_c, train_ds=train_ds, val_ds=val_ds, device="cuda")

    best = float("inf")
    for ep in range(1, _c.epochs + 1):
        tr_loss = trainer.train_one_epoch(train=True)
        va_loss = trainer.train_one_epoch(train=False)

        # optuna reporting + pruning
        trial.report(va_loss, step=ep)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if va_loss < best:
            best = va_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(trainer.model.state_dict(), _c.checkpoint_path)

        #logging.info(f"[DERITS][Epoch {ep}] train: {tr_loss:.4f} | val {va_loss:.4f} | best {best:.4f}")
        logging.info(f"Model: {cfg.model.model_type}, Epoch {ep+1}/{cfg.epochs}: Train Loss={tr_loss:.6f}, Val Loss={va_loss:.6f}")
    logging.info(f"Model: {cfg.model.model_type}, Best Loss:{best:.6f}") 
    return best


def objective(trial, cfg, X_raw, Y_raw, checkpoint_dir):
    # TimesFM dedicated branch
    if cfg.model.trainer_class == 'InTrainer':
        return objective_informer(trial, cfg, X_raw, Y_raw, checkpoint_dir)
    elif cfg.model.trainer_class == 'TimesFMTrainer':
        return objective_timesfm(trial, cfg, X_raw, Y_raw, checkpoint_dir)
    elif cfg.model.trainer_class == 'DERTrainer':
        return object_derits(trial, cfg, X_raw, Y_raw, checkpoint_dir)

    # Define hyperparameter search space with fallback
    batch_size_choices = [8, 16, 32]
    try:
        batch_size = trial.suggest_categorical('batch_size', batch_size_choices)
    except Exception as e:
        logging.warning(f"Batch size sampling failed: {str(e)}. Falling back to default batch_size=16")
        batch_size = 16

    params = {
        'input_size': X_raw.shape[1],  # Dynamically set from X_raw
        'output_size': Y_raw.shape[1],
        'output_len': int(cfg.model.params.get('output_len', 5)),
        'input_len': trial.suggest_int('input_len', 32, 64, step=16),
        'hidden_size': trial.suggest_int('hidden_size', 32, 256, step=16),
        'num_layers': trial.suggest_int('num_layers', 1, 8),
        'batch_size': batch_size,
        'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    }
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    lr_policy = trial.suggest_categorical('lr_policy', ['constant', 'clr', 'onecycle'])

    if cfg.model.trainer_class in ['CNNLSTMTrainer', 'TCNTrainer']:
        params['channels'] = trial.suggest_int('channels', 32, 256, step=32)
        params['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 7])
        params['num_conv_layers'] = trial.suggest_int('num_conv_layers', 1, 8)
    if cfg.model.trainer_class == 'TCNTrainer':
        params['blocks'] = trial.suggest_int('blocks', 2, 6)
    if cfg.model.trainer_class in ['FEDTrainer', 'TSTPTrainer']:
        params['decom_kernel_size'] = trial.suggest_int('decom_kernel_size', 5, 49, step=2)#All Odds(5,7,9,11,...)
        params['top_k_modes'] = trial.suggest_int('top_k_modes', 3, 10)
        params['nhead'] = trial.suggest_categorical('nhead', [4, 8, 16])
        params['dim_feedforward'] = trial.suggest_int('dim_feedforward', 128, 1024, step=64)

    cfg.model.params.update(params)
    cfg.model.lr = lr
    cfg.model.optimizer = optimizer
    cfg.model.lr_policy = lr_policy 

    logging.info(f"[{cfg.model.model_type}] Trial {trial.number} started with params: {params}, lr={lr}, optimizer={optimizer}, lr_policy={lr_policy}")

    clean_memory()
    pre_mem = monitor_memory()

    try:
        x_nans = np.isnan(X_raw).sum()
        y_nans = np.isnan(Y_raw).sum()
        logging.info(f"Trial {trial.number}: X_raw NaNs={x_nans}, Y_raw NaNs={y_nans}, input_size={params['input_size']}")
        if len(X_raw) < params['input_len'] + params['output_len']:
            logging.warning(f"Trial {trial.number}: Insufficient data length {len(X_raw)} for input_len={params['input_len']} + output_len={params['output_len']}")
            raise optuna.TrialPruned()

        dataset = WindowDataset(X_raw, Y_raw, input_len=params['input_len'], output_len=params['output_len'])
        Xw, Yw = dataset.x, dataset.y
        logging.info(f"Trial {trial.number} windowed shapes: Xw={Xw.shape}, Yw={Yw.shape}")

        dataset = TensorDataset(torch.tensor(Xw, dtype=torch.float32), torch.tensor(Yw, dtype=torch.float32))
        N = len(dataset)
        n_val = max(1, int(0.15 * N))
        n_test = max(1, int(0.1 * N))
        n_train = N - n_val - n_test
        train_dataset, val_dataset, _ = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_informer)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_informer)

        trainer_class = TRAINER_CLASSES[cfg.model.trainer_class]
        train_cfg = TrainConfig(**cfg.model.params, 
                                lr=cfg.model.lr, 
                                optimizer=cfg.model.optimizer, 
                                model_type=cfg.model.model_type, 
                                optuna_trial=trial) 
        logging.info(f"Creating trainer with device={train_cfg.device}")
        trainer = trainer_class(train_cfg)

        # Access underlying TensorDataset
        train_tensors = train_dataset.dataset.tensors
        logging.info(f"Trial {trial.number}: Training data shapes: X={train_tensors[0].shape}, Y={train_tensors[1].shape}")
        val_loss, loss_tracker = trainer.train(train_tensors[0].numpy(), train_tensors[1].numpy())
        trainer.save_best_model(
            os.path.join(
                checkpoint_dir,
                f"{cfg.model.model_type}_trial{trial.number}_{val_loss:.6f}.pt"))  #datetime.now().strftime('%Y%m%d_%H%M%S')
        return val_loss

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"Trial {trial.number} failed: GPU Out of Memory Error: {str(e)}\n{traceback.format_exc()}")
        if params['batch_size'] > 16:
            trial.set_user_attr('retry_batch_size', params['batch_size'] // 2)
            trial.set_user_attr('retry_hidden_size', params.get('hidden_size', 128) // 2)
            logging.info(f"Retrying with batch_size={params['batch_size'] // 2}, hidden_size={params.get('hidden_size', 128) // 2}")
            raise optuna.TrialPruned()
        raise
    except Exception as e:
        logging.error(f"Trial {trial.number} failed: {str(e)}\n{traceback.format_exc()}")
        raise optuna.TrialPruned()
    finally:
        # Clean up accelerator state and ensure no lingering tensors
        from accelerate.state import AcceleratorState
        AcceleratorState._reset_state()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        # Delete large variables to free memory
        try:
            del dataset, train_dataset, val_dataset, train_loader, val_loader
        except:
            pass
        post_mem = monitor_memory()
        clean_memory()

# Main orchestration function
def orchestrate_training(cfg: DictConfig):
    # Here cfg is not model-specific config. It is /configs/config.yaml
    set_seed()
    os.makedirs(cfg.experiment.checkpoint_dir, exist_ok=True)

    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Device count: {torch.cuda.device_count()}")
        logging.info(f"Device name: {torch.cuda.get_device_name(0)}")
        os.system('nvidia-smi')
    else:
        logging.error("CUDA not available")

    # Load and preprocess data
    try:
        data_preparer = dataprep(cfg.experiment.data_path)
        data_preparer.one_shot_prep() 
        X_raw = data_preparer.X_full_np  # (N, F)
        Y_raw = data_preparer.Y_full_np  # (N, T)
        if X_raw is None or Y_raw is None:
            raise ValueError("Data Loading Failed - X_full_np or Y_full_np is None")
        logging.info(f"Raw data shapes: X={X_raw.shape}, Y={Y_raw.shape}")
        # Validate data shapes
        batch_x, batch_y = X_raw[:64], Y_raw[:64]
        logging.info(f"Sample batch shapes: X={batch_x.shape}, Y={batch_y.shape}")
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

    # Loop over models
    model_configs = [OmegaConf.load(f"../configs/model/{model}.yaml") for model in cfg.experiment.models]
    for model_cfg in model_configs:
        study = optuna.create_study(
            study_name=f"{model_cfg.model.model_type}_hpo",
            direction=cfg.experiment.optuna.direction,
            storage='sqlite:///optuna_study.db',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20)
        )
        study.optimize(
            lambda trial: objective(trial, OmegaConf.create(model_cfg), X_raw, Y_raw, cfg.experiment.checkpoint_dir),
            n_trials=cfg.experiment.optuna.n_trials,
            n_jobs=cfg.experiment.optuna.n_jobs,
            timeout=3600
        )
        if study.trials and any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
            logging.info(f"Best trial for {model_cfg.model.model_type}: {study.best_trial.number}, value: {study.best_value}, params: {study.best_params}")
        else:
            logging.warning(f"No completed trials for {model_cfg.model.model_type}. All trials were pruned or failed. Trials: {len(study.trials)}")
            logging.info(f"Trial states: {[trial.state for trial in study.trials]}")

# @hydra.main decorator
@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logging.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    orchestrate_training(cfg)

if __name__ == "__main__":
    main()