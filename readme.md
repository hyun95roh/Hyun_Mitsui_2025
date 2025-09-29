## 2025 Mitsui Commodity Price Prediction 

### How to run 
Run the script with

```bash
cd src
python study.py
```

If you want, you can override configs:

```bash
python study.py model=lstm experiment.optuna.n_trials=20
```

You can try different models:

``` bash
python study.py model=transformer
```

For managing multiple model architectures:
1. Add configs like ```configs/model/transformer.yaml```
2. Update MODEL_CLASSES and TRAINER_CLASSES in study.py 
3. Run separate studies for each model:
```bash
python study.py model=lstm experiment.optuna.study_name=lstm_hpo
python study.py model=transformer experiment.optuna.study_name=transformer_hpo
```

### Start testing with small subset 
Choose 10% of rows and 20 targets(out of 400+) to balance speed and representativeness. 

Validate Configs: 
```bash
python -c "from utils import validate_all_configs; validate_all_configs('configs/model/')"
```

Test Subset:
```bash
python src/start_small.py
```

Run Full Pipeline:
```bash
cd src 
python study.py -m model=lstm, cnnlstm, tcn, fed 
```