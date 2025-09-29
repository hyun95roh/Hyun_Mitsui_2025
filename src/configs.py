from dataclasses import dataclass
import torch 

############################
# Classes: 
############################
@dataclass
# output_len = k for k-step ahead.
# lag controls the offset between the date_id of the last input and date_id of the target
# lag adjusts the input data window used for prediction, not the model's output structure. 

class TrainConfig:
    input_len: int = 64  # feature timesteps for windows(len = timesteps) 
    input_size: int = 559  # size = number of columns 
    output_len: int = 1    # predicted timesteps in a single forward pass of model
    output_size: int = 424 # number of targets 
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
    optimizer: str = "AdamW"
    lr_policy: str = "constant"  # constant, clr, onecycle
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    clip_grad_norm: float = 1.0
    partially_finite_target: bool = False
    kernel_size: int = 5 # For CNNLSTM 
    num_conv_layers: int = 2 # For CNNLSTM 
    tcn_kernel: int = 3  # For TCN
    decom_kernel_size: int = 25  # For FED
    top_k_modes: int = 5  # For FED
    nhead: int = 8  # For FED
    dim_feedforward: int = 512  # For FED



############################
# Functions:
############################
# Generate .yaml files dynamically using Python (e.g., for hyperparameter sweeps or new models).
import os
import yaml
from itertools import product

def generate_model_configs(model_name: str, model_class: str, base_params: dict, sweep_params: dict, output_dir: str):
    """
    Generate YAML configs for a model with hyperparameter sweeps.
    
    Args:
        model_name: Base name (e.g., 'FED_Forecaster').
        model_class: Trainer class (e.g., 'FEDTrainer').
        base_params: Base configuration dictionary.
        sweep_params: Dict of params to sweep (e.g., {'hidden_size': [64, 128]}).
        output_dir: Directory to save YAML files (e.g., 'configs/model/').
    """
    os.makedirs(output_dir, exist_ok=True)
    param_combinations = list(product(*[[(k, v) for v in values] for k, values in sweep_params.items()]))
    
    for i, combo in enumerate(param_combinations):
        config = base_params.copy()
        config['model_name'] = f"{model_name}_sweep_{i}"
        config['model_class'] = model_class
        params = config.get('params', {})
        for param, value in combo:
            params[param] = value
        config['params'] = params
        
        with open(os.path.join(output_dir, f"{model_name.lower()}_sweep_{i}.yaml"), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

# Example usage
if __name__ == "__main__":
    base_params = {
        'params': {
            'input_size': 10,
            'output_size': 1,
            'batch_size': 32,
            'dropout': 0.1,
            'num_layers': 2,
            'decom_kernel_size': 25,
            'top_k_modes': 5,
            'nhead': 8,
            'dim_feedforward': 512
        },
        'optimizer': 'Adam',
        'lr': 0.001,
        'epochs': 50,
        'lr_policy': 'constant',
        'patience': 5
    }
    sweep_params = {
        'hidden_size': [64, 128, 256],
        'num_layers': [1, 2],
        'top_k_modes': [3, 5, 7]
    }
    generate_model_configs('FED_Forecaster', 'FEDTrainer', base_params, sweep_params, 'configs/model/')
