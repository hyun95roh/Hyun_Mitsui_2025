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

# Import your modules
from dataprep import dataprep 
from configs import TrainConfig
from models import LSTMForecaster, CNNLSTMForecaster, TCNForecaster, FEDForecaster
from adapters import FEDAdapter, LSTMAdapter, CNNLSTMAdapter, TCNAdapter, TSTPAdapter
from trainer import LSTMTrainer, CNNLSTMTrainer, FEDTrainer, TCNTrainer, TSTPTrainer
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction


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
    'FED': FEDForecaster,  # Add FEDForecaster
    'TFTP': TimeSeriesTransformerForPrediction 
}

TRAINER_CLASSES = {
    'LSTMTrainer': LSTMTrainer,
    'CNNLSTMTrainer': CNNLSTMTrainer,
    'FEDTrainer': FEDTrainer,  # Add FEDTrainer
    'TFTPTrainer': TSTPTrainer
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

# Function to monitor memory usage
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
def objective(trial, cfg, train_loader, val_loader, checkpoint_dir):
    # Define hyperparameter search space
    params = {
        'input_size': cfg.model.params.input_size,
        'output_size': cfg.model.params.output_size,
        'input_len': trial.suggest_int('input_len', 32, 256, step=16),
        'hidden_size': trial.suggest_int('hidden_size', 32, 256, step=16),
        'num_layers': trial.suggest_int('num_layers', 1, 6),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.1)
    }
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD'])
    lr_policy = trial.suggest_categorical('lr_policy', ['constant', 'clr', 'onecycle'])

    # Model-specific hyperparameters
    if cfg.model.model_class in ['CNNLSTMTrainer', 'TCNTrainer']:
        params['channels'] = trial.suggest_int('channels', 32, 256, step=32)
        params['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 7])
        params['num_conv_layers'] = trial.suggest_int('num_conv_layers', 1, 4)
    if cfg.model.model_class == 'TCNTrainer':
        params['blocks'] = trial.suggest_int('blocks', 2, 6)
    if cfg.model.model_class in ['FEDTrainer', 'TransformerTrainer']:
        params['decom_kernel_size'] = trial.suggest_int('decom_kernel_size', 5, 50, step=5)
        params['top_k_modes'] = trial.suggest_int('top_k_modes', 3, 10)
        params['nhead'] = trial.suggest_categorical('nhead', [4, 8, 16])
        params['dim_feedforward'] = trial.suggest_int('dim_feedforward', 128, 1024, step=64)

    cfg.model.params.update(params)
    cfg.model.lr = lr
    cfg.model.optimizer = optimizer
    cfg.model.lr_policy = lr_policy 

    logging.info(f"Trial {trial.number}: {params}, \
                 lr={lr}, \
                 optimizer={optimizer}")

    clean_memory()
    pre_mem = monitor_memory()

    try:
        trainer_class = TRAINER_CLASSES[cfg.model.model_class]
        train_cfg = TrainConfig(**cfg.model.params, lr=cfg.model.lr, optimizer=cfg.model.optimizer)
        trainer = trainer_class(train_cfg, input_size=params['input_size'], num_targets=params['output_size'])

        val_loss = trainer.train(train_loader.dataset.tensors[0].numpy(), train_loader.dataset.tensors[1].numpy())
        trainer.save_best_model(
            os.path.join(
                checkpoint_dir, \
                f"{cfg.model.model_name}_trial{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"))
        return val_loss

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"Trial {trial.number} failed: GPU Out of Memory Error: {str(e)}")
        if params['batch_size'] > 16:
            trial.set_user_attr('retry_batch_size', params['batch_size'] // 2)
            raise optuna.TrialPruned()
        raise
    except Exception as e:
        logging.error(f"Trial {trial.number} failed: {str(e)}")
        raise optuna.TrialPruned()
    finally:
        post_mem = monitor_memory()
        clean_memory()


# Main orchestration function
def orchestrate_training(cfg: DictConfig):
    set_seed()
    os.makedirs(cfg.experiment.checkpoint_dir, exist_ok=True)

    # Load and preprocess data
    try:
        data_preparer = dataprep(cfg.experiment.data_path)  # Pass data_path from config 
        train_loader, val_loader, test_loader = data_preparer.load_and_preprocess_data(
            cfg.experiment.data_path, 
            cfg.model,
            subset = {'num_targets': cfg.experiment.num_targets},
            filter_features = cfg.experiment.filter_features
            )
        logging.info("Data loaded and preprocessed successfully.") 
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)

    # Loop over models
    model_configs = [OmegaConf.load(f"configs/model/{model}.yaml") for model in cfg.experiment.models]
    for model_cfg in model_configs:
        study = optuna.create_study(
            study_name = f"{model_cfg.model_name}_hpo",
            direction = cfg.experiment.optuna.direction,
            storage = 'sqlite:///optuna_study.db',
            load_if_exists = True,
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
        )
        study.optimize(
            lambda trial: objective(trial, OmegaConf.create({'model': model_cfg}), train_loader, val_loader, cfg.experiment.checkpoint_dir),
            n_trials=cfg.experiment.optuna.n_trials,
            n_jobs=cfg.experiment.optuna.n_jobs
        )
        logging.info(f"Best trial for \
                    {model_cfg.model_name}: {study.best_trial.number}, \
                    value: {study.best_value}, \
                    params: {study.best_params}")
        

# @hydra.main decorator - it loads configs 
## Hydra creates a unique output directory per run
@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    # Log resolved config
    logging.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    orchestrate_training(cfg)

if __name__ == "__main__":
    main()