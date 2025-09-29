import pandas as pd
import torch
import logging, time 
import sys, os, traceback
from configs import TrainConfig
from trainer import LSTMTrainer, CNNLSTMTrainer, TCNTrainer, FEDTrainer
from dataprep import dataprep
from adapters import LSTMAdapter, CNNLSTMAdapter, TCNAdapter, FEDAdapter
from omegaconf import OmegaConf


# Set up logging
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'start_small.log')
cli_log_file = os.path.join(log_dir, 'start_small_cli.log')

# Clear log files
try:
    with open(log_file, 'w') as f:
        f.write('')
    with open(cli_log_file, 'w') as f:
        f.write('')
except PermissionError as e:
    print(f"PermissionError: Cannot write to log files: {e}")
    raise

# Custom stream handler to redirect print to CLI log file
class StreamToLogger:
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
        with open(cli_log_file, 'a') as f:
            f.write(buf)

    def flush(self):
        pass

logging.basicConfig(
    level=logging.DEBUG, #logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger()
sys.stdout = StreamToLogger(logger, logging.INFO)

# Define trainer and adapter mappings
TRAINER_CLASSES = {
    'LSTM': LSTMTrainer,
    'CNNLSTM': CNNLSTMTrainer,
    'TCN': TCNTrainer,
    'FED': FEDTrainer
}

ADAPTER_CLASSES = {
    'LSTMTrainer': LSTMAdapter,
    'CNNLSTMTrainer': CNNLSTMAdapter,
    'TCNTrainer': TCNAdapter,
    'FEDTrainer': FEDAdapter
}

# Models to test
MODELS = ['lstm', 'cnnlstm', 'tcn', 'fed']
print(f"Models to test: {MODELS}")

# Define data path
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Test both feature filtering modes
for filter_features in [False]:  # If True, then sort out only relative features for target pair.
    logging.info(f"Testing with filter_features={filter_features}")
    # Data prep with subset
    data_preparer = dataprep(data_path)
    try:
        base_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml'))
        data_cfg = TrainConfig(input_len=64, output_len=1, batch_size=16)
        train_loader, val_loader, test_loader = data_preparer.load_and_preprocess_data(
            data_path=data_path,
            cfg=data_cfg,
            subset={'row_fraction': 0.1, 'num_targets': 20},
            filter_features=filter_features
        )
        batch_x, batch_y = next(iter(train_loader))
        logging.info(f"Train loader batch shapes (filter_features={filter_features}): X={batch_x.shape}, Y={batch_y.shape}")
    except Exception as e:
        logging.error(f"Data loading failed (filter_features={filter_features}): {str(e)}")
        raise

    # Iterate over models
    for model_name in MODELS:
        logging.info(f"Testing model: {model_name}")
        try:
            model_cfg = OmegaConf.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'model', f'{model_name}.yaml'))
            config_params = {
                'input_len': model_cfg.params.get('input_len', 64),
                'output_len': model_cfg.params.get('output_len', 1),
                'batch_size': model_cfg.params.get('batch_size', 16),
                'hidden_size': model_cfg.params.hidden_size,
                'num_layers': model_cfg.params.num_layers,
                'dropout': model_cfg.params.dropout,
                'epochs': 5,
                'patience': 3
            }
            if model_name in ['cnnlstm', 'tcn']:
                config_params['channels'] = model_cfg.params.get('channels', 64)
                config_params['blocks'] = model_cfg.params.get('blocks', 3)
            if model_name == 'tcn':
                config_params['tcn_kernel'] = model_cfg.params.get('tcn_kernel', 3)
            if model_name == 'fed':
                config_params['decom_kernel_size'] = model_cfg.params.get('decom_kernel_size', 25)
                config_params['top_k_modes'] = model_cfg.params.get('top_k_modes', 5)
                config_params['nhead'] = model_cfg.params.get('nhead', 8)
                config_params['dim_feedforward'] = model_cfg.params.get('dim_feedforward', 512)

            cfg = TrainConfig(**config_params)
            cfg.input_size = batch_x.shape[2]  # Dynamic feature count
            cfg.output_size = batch_y.shape[2]
            logging.info(f"Config for {model_name}: input_len={cfg.input_len}, input_size={cfg.input_size}, output_size={cfg.output_size}, output_len={cfg.output_len}")

            # Train
            trainer_class = TRAINER_CLASSES[model_cfg.model_type]
            trainer = trainer_class(cfg)  # Pass only cfg
            X_train = train_loader.dataset.dataset.tensors[0].numpy()
            Y_train = train_loader.dataset.dataset.tensors[1].numpy()
            logging.info(f"Training input shapes for {model_name} (filter_features={filter_features}): X_train={X_train.shape}, Y_train={Y_train.shape}")
            val_loss = trainer.train(X_train, Y_train)
            logging.info(f"Validation loss for {model_name} (filter_features={filter_features}): {val_loss}")

            # Test prediction with different lags
            if not isinstance(data_preparer.Y_full, pd.DataFrame):
                raise ValueError("Y_full is not a DataFrame")
            logging.info(f"Y_full shape: {data_preparer.Y_full.shape}")
            target_cols = list(data_preparer.Y_full.columns[:cfg.output_size])
            if len(target_cols) != cfg.output_size:
                logging.error(f"Target columns mismatch for {model_name}: expected {cfg.output_size}, got {len(target_cols)}")
                raise ValueError(f"Target columns mismatch: {target_cols}")
            list(data_preparer.Y_full.columns[:cfg.output_size])
            logging.info(f"Target columns for {model_name} (filter_features={filter_features}): {target_cols}")
            adapter_class = ADAPTER_CLASSES[model_cfg.trainer_class]
            adapter = adapter_class(trainer.model, trainer.x_scaler, target_cols=target_cols)
            logging.info(f"Model device for {model_name}: {next(trainer.model.parameters()).device}")

            for lag in [1, 2, 3, 4]:
                logging.info(f"Testing lag={lag} for {model_name} (filter_features={filter_features})")
                X_hist = data_preparer.X_full.iloc[-cfg.input_len-lag:].copy()
                Y_hist = data_preparer.Y_full.iloc[-cfg.input_len-lag:].copy()
                try:
                    start_time = time.time()
                    pred, metadata = adapter.predict_next(X_hist, Y_hist, cfg.input_len, lag=lag)
                    logging.info(f"Prediction --- done! | Duration: {time.time() - start_time}")
                    logging.info(f"Prediction shape for {model_name} (lag={lag}, filter_features={filter_features}): {pred.shape}")
                    logging.info(f"Prediction metadata for {model_name} (lag={lag}, filter_features={filter_features}): {metadata}")
                    if metadata['date_id'] is not None and metadata['target_date_id'] is not None:
                        expected_target_date_id = X_hist['date_id'].iloc[-1] + lag + 1
                        if metadata['target_date_id'] != expected_target_date_id:
                            logging.warning(f"Lag validation failed for {model_name}: expected target_date_id={expected_target_date_id}, got {metadata['target_date_id']}")
                except Exception as e:
                    logging.error(f"Prediction failed for {model_name} (lag={lag}, filter_features={filter_features}): {str(e)}\n{traceback.format_exc()}")
                    continue

        except Exception as e:
            logging.error(f"Failed for model {model_name} (filter_features={filter_features}): {str(e)}")
            continue