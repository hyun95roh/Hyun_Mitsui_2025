import os, sys
import logging, traceback 
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
import gc
import hydra
from omegaconf import DictConfig, OmegaConf
import optuna
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from dataprep import WindowDataset
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction as TSTP
# Import your modules
from dataprep import dataprep, SeriesSpec, TimesFMSingleSeries, split_by_windows
from configs import TrainConfig
from models import LSTMForecaster, CNNLSTMForecaster, TCNForecaster, FEDForecaster
from adapters import FEDAdapter, LSTMAdapter, CNNLSTMAdapter, TCNAdapter, TSTPAdapter
from trainer import LSTMTrainer, CNNLSTMTrainer, FEDTrainer, TCNTrainer, TSTPTrainer, TimesFMTrainer


# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
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
    'TSTP': TSTP
}

TRAINER_CLASSES = {
    'LSTMTrainer': LSTMTrainer,
    'CNNLSTMTrainer': CNNLSTMTrainer,
    'FEDTrainer': FEDTrainer,
    'TCNTrainer': TCNTrainer,
    'TSTPTrainer': TSTPTrainer
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
def objective(trial, cfg, X_raw, Y_raw, checkpoint_dir):
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
        'output_len': 1, 
        'input_len': trial.suggest_int('input_len', 32, 64, step=16),
        'hidden_size': trial.suggest_int('hidden_size', 32, 128, step=16),
        'num_layers': trial.suggest_int('num_layers', 1, 4),
        'batch_size': batch_size,
        'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    }
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    lr_policy = trial.suggest_categorical('lr_policy', ['constant', 'clr', 'onecycle'])

    if cfg.model.trainer_class in ['CNNLSTMTrainer', 'TCNTrainer']:
        params['channels'] = trial.suggest_int('channels', 32, 256, step=32)
        params['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 7])
        params['num_conv_layers'] = trial.suggest_int('num_conv_layers', 1, 4)
    if cfg.model.trainer_class == 'TCNTrainer':
        params['blocks'] = trial.suggest_int('blocks', 2, 6)
    if cfg.model.trainer_class in ['FEDTrainer', 'TSTPTrainer']:
        params['decom_kernel_size'] = trial.suggest_int('decom_kernel_size', 5, 50, step=5)
        params['top_k_modes'] = trial.suggest_int('top_k_modes', 3, 10)
        params['nhead'] = trial.suggest_categorical('nhead', [4, 8, 16])
        params['dim_feedforward'] = trial.suggest_int('dim_feedforward', 128, 1024, step=64)

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
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)

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
                f"{cfg.model.model_type}_trial{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"))
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
    set_seed()
    os.makedirs(cfg.experiment.checkpoint_dir, exist_ok=True)

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
            study_name=f"{model_cfg.model_type}_hpo",
            direction=cfg.experiment.optuna.direction,
            storage='sqlite:///optuna_study.db',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20)
        )
        study.optimize(
            lambda trial: objective(trial, OmegaConf.create({'model': model_cfg}), X_raw, Y_raw, cfg.experiment.checkpoint_dir),
            n_trials=cfg.experiment.optuna.n_trials,
            n_jobs=cfg.experiment.optuna.n_jobs,
            timeout=3600
        )
        if study.trials and any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
            logging.info(f"Best trial for {model_cfg.model_type}: {study.best_trial.number}, value: {study.best_value}, params: {study.best_params}")
        else:
            logging.warning(f"No completed trials for {model_cfg.model_type}. All trials were pruned or failed. Trials: {len(study.trials)}")
            logging.info(f"Trial states: {[trial.state for trial in study.trials]}")

# @hydra.main decorator
@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    logging.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    orchestrate_training(cfg)

if __name__ == "__main__":
    main()