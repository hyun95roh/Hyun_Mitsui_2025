# From sibiling module
from configs import TrainConfig
from utils import log_returns

import pandas as pd 
import numpy as np 
import torch 
import os 
import logging, traceback
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, List, Dict, Callable, Optional, Protocol, Iterable
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature, AddObservedValuesIndicator, AddTimeFeatures,
    AsNumpyArray, Chain, ExpectedNumInstanceSampler, RemoveFields,
    SelectFields, SetField, TestSplitSampler, Transformation, ValidationSplitSampler, VstackFeatures
)
from gluonts.time_feature import time_features_from_frequency_str

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

######################
# Dataset
######################
class WindowDataset(Dataset):
    """
    General windower:
      X_full: (N, F) if multivariate, None if univariate (use Y_full)
      Y_full: (N,) for univariate -> promoted to (N, 1)
              (N, T) for multi-target
    Outputs:
      x: (N_windows, input_len, F)
      y: (N_windows, output_len, num_targets)
    """
    def __init__(self, X_full: Optional[np.ndarray], Y_full: np.ndarray, input_len: int = 64, output_len: int = 1):
        Y_full = np.asarray(Y_full, dtype=np.float32)
        if Y_full.ndim == 1:
            Y_full = Y_full[:, None]
        if X_full is None:
            X_full = Y_full.astype(np.float32)
        else:
            X_full = np.asarray(X_full, dtype=np.float32)
        if X_full.ndim != 2 or Y_full.ndim != 2:
            raise ValueError(f"Expected X_full (N, F) and Y_full (N, T), got shapes {X_full.shape}, {Y_full.shape}")
        self.x, self.y = [], []
        N = len(Y_full)
        invalid_windows = 0
        for i in range(N - input_len - output_len + 1):
            xw = X_full[i:i+input_len]
            yw = Y_full[i+input_len:i+input_len+output_len]
            if np.all(np.isfinite(xw)) and np.all(np.isfinite(yw)):
                self.x.append(xw)
                self.y.append(yw)
            else:
                invalid_windows += 1
        self.x = np.asarray(self.x, dtype=np.float32)
        self.y = np.asarray(self.y, dtype=np.float32)
        if len(self.x) == 0:
            logging.error(f"No valid windows created. Input length: {N}, input_len: {input_len}, output_len: {output_len}, invalid_windows: {invalid_windows}")
            raise ValueError(f"No valid windows created. Input length: {N}, input_len: {input_len}, output_len: {output_len}")
        logging.info(f"WindowDataset created: Xw shape={self.x.shape}, Yw shape={self.y.shape}, invalid_windows: {invalid_windows}")

    def __len__(self): return len(self.x)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

def parse_pair_field(pair:str) -> Tuple[str, Optional[str]]:
    """
    Parse the 'pair' field from target_pair.csv
    """
    parts = [p.strip() for p in pair.split('-')] 
    if len(parts) == 1:
        return parts[0], None 
    elif len(parts) == 2:
        return parts[0], parts[1] 
    else:
        # Defensive: if more hyphens appear inside names
        return parts[0], "-".join(parts[1:]).strip() 

def impute_features_with_staleness(df: pd.DataFrame, date_col="date_id", cap_days: int | None = None):
    """
    Returns:
      X_imputed: DataFrame of features (float), LOCF-then-backfill
      X_stale:   DataFrame (same shape) with 'days since last observed' (0 if observed today)
      X_miss:    DataFrame of 0/1 missing mask (1 if originally missing)
    Notes:
      - Does not touch `date_col`.
      - cap_days: if set, forward-fill only up to cap; beyond cap keep NaN.
    """
    idxed = df.set_index(date_col).sort_index()
    feats = idxed.select_dtypes(include=[np.number, "float", "int"]).copy()

    # original missing mask
    miss = feats.isna().astype(np.float32)

    # compute "days since last observed" staleness
    # approach: forward-fill an index counter of observed rows per column
    obs_idx = (~feats.isna()).astype(int)
    # cumulative count of observed points per column
    csum = obs_idx.cumsum()
    # for missing entries, staleness = (current row index) - (row index of last observation)
    # we can approximate using csum differences:
    # Build a per-column "last seen position" by forward-filling row numbers where obs occurred
    rownum = pd.Series(np.arange(len(feats)), index=feats.index)
    last_seen = feats.notna().mul(rownum, axis=0).where(feats.notna())
    last_seen = last_seen.ffill()
    staleness = (rownum.values.reshape(-1,1) - last_seen.values).astype("float")
    staleness_df = pd.DataFrame(staleness, index=feats.index, columns=feats.columns).fillna(0.0)

    # forward-fill
    ffilled = feats.ffill(limit=cap_days)
    # back-fill the head so first windows can form
    bfilled = ffilled.bfill()

    return (
        bfilled.reset_index(),                                 # X_imputed
        staleness_df.reset_index(),                            # X_stale
        miss.reset_index(),                                    # X_miss
    )

def create_gluonts_transformation(config: dict, freq: str = 'D'):
    """GluonTS transformation chain for multivariate time series with shared features."""
    remove_fields = [
        FieldName.FEAT_STATIC_REAL,
        FieldName.FEAT_STATIC_CAT,  # No static features in competition data
    ]

    time_features = time_features_from_frequency_str(freq)  # e.g., [year, month, day, dayofweek]
    transformation = Chain([
        RemoveFields(remove_fields),
        AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,  # Binary mask for missing values
            imputation_method=None  # Use default (mean) imputation for NaNs
        ),
        AddTimeFeatures(
            start_field=FieldName.START,
            target_field=FieldName.TARGET,
            output_field=FieldName.FEAT_TIME,  # Store time features
            time_features=time_features,
            pred_length=config['prediction_length']
        ),
        VstackFeatures(
            input_fields=[FieldName.FEAT_DYNAMIC_REAL, FieldName.FEAT_TIME],  # Stack dynamic + time features
            output_field=FieldName.FEAT_DYNAMIC_REAL  # Overwrite dynamic real with stacked features
        ),
        AsNumpyArray(
            field=FieldName.FEAT_DYNAMIC_REAL,
            expected_ndim=2,  # (T, F + time_features)
            dtype=np.float32
        ),
        AsNumpyArray(
            field=FieldName.TARGET,
            expected_ndim=2,  # (T, 424) for multivariate targets
            dtype=np.float32
        ),
    ])
    return transformation

def generate_single_target(
        prices_idxed: pd.DataFrame, 
        asset_a: str|None, 
        asset_b: Optional[str], 
        lag: int 
) -> pd.Series: 
    """ 
    Produce one target series per competition rule:
        target = log_return(asset_a, lag) - log_return(asset_b, lag) if asset_b else log_return(asset_a, lag)
    Result is indexed by date_id (same as prices_idxed[date_col]). 
    """
    if asset_a not in prices_idxed.columns:
        raise KeyError(f"Asset column [{asset_a}] not found in columns")
    
    ret_a = log_returns(prices_idxed[asset_a], lag) 
    if asset_b is None:
        target = ret_a 
        
    else:
        if asset_b not in prices_idxed.columns:
            raise KeyError(f"Asset column [{asset_a}] not found in columns")
        ret_b = log_returns(prices_idxed[asset_b], lag)
        target = ret_a - ret_b 
    
    target.name = f"{asset_a}" if asset_b is None else f"{asset_a} - {asset_b}"

    assert isinstance(target, pd.Series), "Target is not a pandas Series"
    assert target.index.equals(prices_idxed.index), "target index mismatch with date_id index"
    return target 

def build_all_targets(
        prices_df: pd.DataFrame, 
        target_pairs_df: pd.DataFrame, 
        date_col: str = "date_id", 
        id_col: str = "target", 
) -> pd.DataFrame:
    """
    Build a wide DataFrame of all targets from target_pairs.csv. 
    - prices_df: contains date_id and asset price columns 
    - target_pairs_df: has columns ['target', 'lag', 'pair']

    Returns:
     targets_df with index = date_id, 
     columns = [target_0, ...]
    """
    # work with an indexed view
    prices_idxed = prices_df.set_index(date_col).sort_index()

    series_list = []   # collect Series
    col_names   = []   # and their intended column names (target_0, target_1, ...)

    for _, row in target_pairs_df.iterrows():
        tname = str(row[id_col])
        lag   = int(row["lag"])
        pair  = str(row["pair"])

        # robust split (handles spaces)
        parts = [p.strip() for p in pair.split("-")]
        if len(parts) == 1:
            a, b = parts[0], None
        elif len(parts) == 2:
            a, b = parts[0], parts[1]
        else:
            a, b = parts[0], "-".join(parts[1:]).strip()

        target_series = generate_single_target(prices_idxed, a, b, lag)
        # rename the Series to the competition target name
        target_series = target_series.rename(tname)

        series_list.append(target_series)
        col_names.append(tname)

    # 3) Use concat, not DataFrame(dict)
    targets_df = pd.concat(series_list, axis=1)
    # Ensure column order matches the order we iterated
    targets_df = targets_df.loc[:, col_names]

    # Final tidy: keep date_id as the index
    targets_df.index.name = date_col
    return targets_df

def make_gateway_row(
    preds_by_lag: dict[int, np.ndarray],      # {1: (424,), 2: (424,), 3: (424,), 4: (424,)}
    cols_by_lag: dict[int, list[str]]         # {1: [...424 names...], ...} — names only used to keep order
) -> pd.DataFrame:
    # Ensure order: 1,2,3,4
    row = []
    for lag in (1, 2, 3, 4):
        preds = preds_by_lag[lag]
        assert preds.ndim == 1, f"lag {lag} preds must be (TGT,)"
        assert len(preds) == len(cols_by_lag[lag]), f"lag {lag} length mismatch"
        row.append(preds)
    flat = np.concatenate(row)   # (1696,)

    # Column names aren’t checked by the gateway, but using suffixed names helps debugging
    cols = []
    for lag in (1, 2, 3, 4):
        cols.extend([f"{c}_lag{lag}" for c in cols_by_lag[lag]])

    return pd.DataFrame([flat], columns=cols)   # 1 row, no 'date_id'

def make_submission_frame(
        pred_vector: np.ndarray, 
        target_cols: list[str], 
        next_date_id: int, 
        label_lag: int = 1, 
) -> pd.DataFrame:
    """
    Build a single-row submission: 
      [date_id, target_0, ..., target_423, label_date_id]
    - pred_vector: (Target, ) in the same order as target_cols. 
    - next_date_id: the date_id you're submitting for (usually last_index+1)
    - label_lag: sets label_date_id = next_date_id - (label_lag +1)
    """
    assert len(pred_vector)==len(target_cols), "pred length != number of targets"
    row = [next_date_id] + list(map(float, pred_vector)) + [int(next_date_id - (label_lag +1))]
    cols = ["date_id"] + target_cols + ["label_date_id"] 
    return pd.DataFrame([row], columns=cols) 


##############################
# Dataprep 
##############################
class dataprep:
    # ---- Paths (adjust if needed)
    #PRICES_PATH = "data/train.csv"            # must include 'date_id' + asset columns
    #TP_PATH     = "data/target_pairs.csv"     # columns: target, lag, pair
    #LABELS_PATH = "data/train_labels.csv"     # optional sanity check

    def __init__(self, data_path:str):
        self.data_path = data_path
        self.prices_path = os.path.join(data_path, "train.csv")
        self.tp_path = os.path.join(data_path, "target_pairs.csv")
        self.labels_path = os.path.join(data_path, "train_labels.csv")
        self.prices_df, self.target_pairs_df, self.targets_df = None, None, None 
        self.Xi, self.Xs, self.Xm = None, None, None 
        self.X_full, self.Y_full = None, None 
        self.X_full_np, self.Y_full_np = None, None 
        self.target_pairs_df = None 
        self._prep_done = False 

    def validate_paths(self):
        """Check if data files exist."""
        for path in [self.prices_path, self.tp_path, self.labels_path]:
            if not os.path.exists(path):
                logging.error(f"File not found: {path}")
                raise FileNotFoundError(f"File not found: {path}")


    @staticmethod
    def drop_last_n_rows(df: pd.DataFrame, n):
        rows = len(df) - n
        return df.iloc[:rows, :]

    def load_data(self, skipfooter=90, skiprows=0):
        self.validate_paths()
        self.prices_df = pd.read_csv(self.prices_path, skiprows=skiprows, skipfooter=skipfooter, engine='python').sort_values("date_id").reset_index(drop=True)
        self.target_pairs_df = pd.read_csv(self.tp_path)

    def build_targets(self):
        self.targets_df = build_all_targets(self.prices_df, self.target_pairs_df, date_col="date_id", id_col="target")
        logging.info(f"targets_df shape: {self.targets_df.shape}")
        self.target_cols = self.targets_df.columns

    def impute_features(self):
        feature_cols = [c for c in self.prices_df.columns if c != 'date_id']
        X_imp, X_stale, X_miss = impute_features_with_staleness(
            self.prices_df[['date_id'] + feature_cols],
            date_col='date_id',
            cap_days=None
        )
        self.Xi = X_imp.set_index("date_id")[feature_cols]
        self.Xs = X_stale.set_index("date_id")[feature_cols].add_suffix("_stale")
        self.Xm = X_miss.set_index("date_id")[feature_cols].add_suffix("_miss")
        X_full = pd.concat([self.Xi, self.Xs, self.Xm], axis=1).astype(float)
        self.X_full = X_full.reset_index().rename(columns={'index': 'date_id'})

    def one_shot_prep(self):
        """Load and preprocess data once, storing results."""
        if self._prep_done:
            return
        try:
            # Load data files
            train_path = os.path.join(self.data_path, 'train.csv')
            labels_path = os.path.join(self.data_path, 'train_labels.csv')
            pairs_path = os.path.join(self.data_path, 'target_pairs.csv')

            # Check if files exist
            for path in [train_path, labels_path, pairs_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Data file not found: {path}")

            logging.info(f"Loading train.csv from {train_path}")
            self.X_full = pd.read_csv(train_path)
            logging.info(f"Loading train_labels.csv from {labels_path}")
            self.Y_full = pd.read_csv(labels_path)
            logging.info(f"Loading target_pairs.csv from {pairs_path}")
            self.target_pairs_df = pd.read_csv(pairs_path)

            # Validate data
            if self.X_full.empty:
                raise ValueError("X_full (train.csv) is empty")
            if self.Y_full.empty:
                raise ValueError("Y_full (train_labels.csv) is empty")
            if self.target_pairs_df.empty:
                raise ValueError("target_pairs_df (target_pairs.csv) is empty")

            # Ensure date_id is present
            if 'date_id' not in self.X_full.columns:
                raise ValueError("X_full missing 'date_id' column")
            if 'date_id' not in self.Y_full.columns:
                raise ValueError("Y_full missing 'date_id' column")

            # Impute NaNs in X_full and include staleness/missingness
            X_imputed, X_stale, X_miss = impute_features_with_staleness(self.X_full)
            # Concatenate features
            X_imputed = X_imputed.set_index('date_id')
            X_stale = X_stale.set_index('date_id').add_suffix('_stale')
            X_miss = X_miss.set_index('date_id').add_suffix('_miss')
            self.X_full = pd.concat([X_imputed, X_stale, X_miss], axis=1).reset_index()

            # Impute NaNs in Y_full
            self.Y_full = self.Y_full.set_index('date_id').ffill().bfill().reset_index()

            # Log NaN statistics and feature counts
            x_nans = self.X_full.drop(columns=['date_id']).isna().sum().sum()
            y_nans_before = self.Y_full.drop(columns=['date_id']).isna().sum().sum()
            logging.info(f"X_full NaNs after imputation: {x_nans}")
            logging.info(f"Y_full NaNs before imputation: {y_nans_before}")
            y_nans_after = self.Y_full.drop(columns=['date_id']).isna().sum().sum()
            logging.info(f"Y_full NaNs after imputation: {y_nans_after}")
            logging.info(f"X_full feature count: {len(self.X_full.columns) - 1}")  # Exclude date_id

            # Convert to numpy
            self.X_full_np = self.X_full.drop(columns=['date_id']).to_numpy(dtype=np.float32)
            self.Y_full_np = self.Y_full.drop(columns=['date_id']).to_numpy(dtype=np.float32)

            # Validate numpy arrays
            if self.X_full_np is None or self.Y_full_np is None:
                raise ValueError("Failed to convert X_full or Y_full to numpy arrays")
            if self.X_full_np.shape[0] == 0 or self.Y_full_np.shape[0] == 0:
                raise ValueError(f"Empty numpy arrays: X_full_np={self.X_full_np.shape}, Y_full_np={self.Y_full_np.shape}")

            logging.info(f"Data loaded: X_full_np shape={self.X_full_np.shape}, Y_full_np shape={self.Y_full_np.shape}")
            self._prep_done = True

        except Exception as e:
            logging.error(f"Error in one_shot_prep: {str(e)}\n{traceback.format_exc()}")
            raise


    def preprocess_for_frequency(self, X: np.ndarray, use_fft: bool = False) -> np.ndarray:
        if not use_fft:
            return X
        fft_features = np.abs(np.fft.rfft(X, axis=0))[:min(10, X.shape[0] // 2 + 1), :]
        X_aug = np.concatenate([X, fft_features.T], axis=1)
        return X_aug


    def GluonTS_transform(self, cfg, model_type=['TSTP']):
        """Add time features manually for multivariate time series, bypassing ListDataset."""
        if not hasattr(self, 'X_full_np') or not hasattr(self, 'Y_full_np'):
            self.one_shot_prep()
        X, Y = self.X_full_np, self.Y_full_np  # (N, 559), (N, 424)

        # Add time features (daily: dayofweek, month, quarter, year)
        start_date = pd.Timestamp("2020-01-01")  # Adjust to min(prices_df['date_id']) if known
        dates = pd.date_range(start=start_date, periods=X.shape[0], freq='D')
        time_df = pd.DataFrame(index=dates)
        time_df['dayofweek'] = time_df.index.dayofweek.astype(np.float32)
        time_df['month'] = time_df.index.month.astype(np.float32)
        time_df['quarter'] = time_df.index.quarter.astype(np.float32)
        time_df['year'] = (time_df.index.year - time_df.index.year.min()).astype(np.float32)  # Normalize year
        time_feats = time_df.to_numpy()  # (N, 4)

        # Concatenate time features to X
        X_transformed = np.concatenate([X, time_feats], axis=1)  # (N, 559 + 4)
        Y_transformed = np.nan_to_num(Y, nan=0.0)  # Impute NaNs, align with one_shot_prep

        logging.info(f"Manual time features added: X_transformed shape {X_transformed.shape}, Y_transformed shape {Y_transformed.shape}")
        return X_transformed, Y_transformed


    def load_and_preprocess_data(self, data_path: str, cfg: TrainConfig, subset: dict = None, filter_features: bool = False):
        """
        Load and preprocess data, creating DataLoaders with optional subsetting and feature filtering.
        """
        self.one_shot_prep()

        X = self.X_full_np
        Y = self.Y_full_np

        logging.info(f"Raw X shape: {X.shape}, Y shape: {Y.shape}")

        if subset:
            row_fraction = subset.get('row_fraction', 1.0)
            num_targets = subset.get('num_targets', Y.shape[1])
            n_rows = int(len(X) * row_fraction)
            if n_rows < cfg.input_len + cfg.output_len:
                n_rows = cfg.input_len + cfg.output_len
                logging.warning(f"Requested {row_fraction*100}% rows ({int(len(X) * row_fraction)}), but minimum {n_rows} required for windowing")
            X = X[:n_rows]
            Y = Y[:n_rows, :num_targets]
            logging.info(f"Subset X shape: {X.shape}, Y shape: {Y.shape}")

            if filter_features:
                target_cols = list(self.Y_full.columns[:num_targets])
                feature_cols = set()
                for target in target_cols:
                    pair_row = self.target_pairs_df[self.target_pairs_df['target'] == target]
                    if not pair_row.empty:
                        pair = pair_row['pair'].iloc[0]
                        if '-' in pair:
                            feature1, feature2 = pair.split('-')
                            feature_cols.add(feature1.strip())
                            feature_cols.add(feature2.strip())
                        else:
                            feature_cols.add(pair.strip())
                feature_cols = list(feature_cols)
                all_feature_cols = feature_cols + [f"{c}_stale" for c in feature_cols] + [f"{c}_miss" for c in feature_cols]
                logging.info(f"all_feature_cols example: {all_feature_cols[:5]}")
                logging.info(f"X_full.columns example: {list(self.X_full.columns[:5])}")
                feature_indices = [i for i, col in enumerate(self.X_full.columns) if col in all_feature_cols]
                if not feature_indices:
                    logging.error(f"No matching features found for {all_feature_cols}")
                    raise ValueError(f"No matching features found for {all_feature_cols}")
                X = X[:, feature_indices]
                logging.info(f"Filtered features for {num_targets} targets: {len(feature_cols)} base features, {len(feature_indices)} total, X shape: {X.shape}")

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(f"Expected X (N, F) and Y (N, T), got shapes {X.shape}, {Y.shape}")

        if len(X) < cfg.input_len + cfg.output_len:
            raise ValueError(f"Input length {len(X)} is less than required {cfg.input_len + cfg.output_len} for windowing")

        X = self.preprocess_for_frequency(X, use_fft=False)
        logging.info(f"After preprocess X shape: {X.shape}")

        cfg.input_size = X.shape[1]
        cfg.output_size = Y.shape[1]

        model_name = getattr(cfg, 'model_name', None)
        if model_name is None:
            model_name_map = {
                'LSTMTrainer': 'LSTM',
                'CNNLSTMTrainer': 'CNNLSTM',
                'TCNTrainer': 'TCN',
                'FEDTrainer': 'FED',
                'TSTPTrainer': 'TSTP',
                'TimesFMTrainer': 'TimesFM'
            }
            model_name = model_name_map.get(getattr(cfg, 'model_class', 'default'), 'default')
            logging.info(f"model_name not provided in cfg; inferred as {model_name} from model_class")

        if model_name == 'TSTP':
            transformation = self.create_gluonts_transformation(
                freq='D',
                config={'prediction_length': cfg.output_len, 'num_static_real_features': 0}
            )
            start_date = pd.Timestamp("2020-01-01")
            dataset_list = []
            for i in range(Y.shape[1]):
                target = Y[:, i]
                feat_dynamic_real = X
                dataset_list.append({
                    FieldName.TARGET: target,
                    FieldName.START: start_date,
                    FieldName.FEAT_DYNAMIC_REAL: feat_dynamic_real,
                })
            gluonts_dataset = ListDataset(dataset_list, freq='D')
            transformed_dataset = transformation.apply(gluonts_dataset)
            
            # Split train/val
            N = len(transformed_dataset)
            n_val = max(1, int(0.15 * N))
            n_test = max(1, int(0.1 * N))
            n_train = N - n_val - n_test
            train_data = list(transformed_dataset)[:n_train]
            val_data = list(transformed_dataset)[n_train:n_train+n_val]
            test_data = list(transformed_dataset)[n_train+n_val:]
            
            train_dataset = ListDataset(train_data, freq='D')
            val_dataset = ListDataset(val_data, freq='D')
            test_dataset = ListDataset(test_data, freq='D')
            
            logging.info(f"GluonTS datasets created: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
            return train_dataset, val_dataset, test_dataset
        else:
            try:
                dataset = WindowDataset(X, Y, cfg.input_len, cfg.output_len)
                Xw, Yw = dataset.x, dataset.y
                logging.info(f"Windowed X shape: {Xw.shape}, Y shape: {Yw.shape}")
            except Exception as e:
                logging.error(f"Windowing failed: {str(e)}")
                raise

            dataset = TensorDataset(
                torch.tensor(Xw, dtype=torch.float32),
                torch.tensor(Yw, dtype=torch.float32)
            )

            N = len(dataset)
            n_val = max(1, int(0.15 * N))
            n_test = max(1, int(0.1 * N))
            n_train = N - n_val - n_test

            if n_train < 1:
                raise ValueError(f"Insufficient training samples after split: {n_train}")

            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
            )

            train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

            return train_loader, val_loader, test_loader
        


#############################
# timesfm
#############################
@dataclass
class SeriesSpec:
    series: np.ndarray              # shape [N,] or [N,1]
    context_len: int
    horizon_len: int
    freq_type: int = 0              # 0=daily

class TimesFMSingleSeries(Dataset):
    """
    Convert one univariate series into rolling windows for TimesFM finetune:
      returns (x_context [C,1], x_padding [C], x_future [H,1], freq [])
    """
    def __init__(self, spec: SeriesSpec):
        y = spec.series.astype(np.float32)
        if y.ndim == 1:
            y = y[:, None]
        self.y = y
        self.context_len = int(spec.context_len)
        self.horizon_len = int(spec.horizon_len)
        self.freq_type = int(spec.freq_type)
        self.indices = []
        max_start = len(y) - (self.context_len + self.horizon_len)
        for t in range(max(0, max_start + 1)):
            self.indices.append(t)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        ctx = self.y[t: t+self.context_len]
        fut = self.y[t+self.context_len: t+self.context_len+self.horizon_len]
        x_context = torch.from_numpy(ctx)                            # [C,1]
        x_future  = torch.from_numpy(fut)                            # [H,1]
        x_padding = torch.zeros(self.context_len, dtype=torch.int64) # [C]
        freq      = torch.tensor(self.freq_type, dtype=torch.int64)  # []
        return x_context, x_padding, x_future, freq
    
def split_by_windows(ds: TimesFMSingleSeries, val_frac: float = 0.2) -> Tuple[TimesFMSingleSeries, TimesFMSingleSeries]:
    """Split by window index so both splits have ≥ 1 sample."""
    n = len(ds)
    if n == 0:
        return ds, ds
    n_val = max(1, int(round(n * val_frac)))
    n_tr  = max(1, n - n_val)
    import copy
    tr = copy.copy(ds); va = copy.copy(ds)
    tr.indices = ds.indices[:n_tr]
    va.indices = ds.indices[-n_val:]
    return tr, va