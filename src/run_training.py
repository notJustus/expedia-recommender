import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import sys
import os
import datetime
import json
import lightgbm as lgb # For typing hint
import time # For timing

# Add src directory to Python path
sys.path.append(os.path.dirname(__file__))

# Updated import: load_data from data_loader, and others from preprocessing
from data.data_loader import load_data 
from data.preprocessing import handle_missing_values
from data.feature_engineering import apply_feature_engineering
from models.lambdamart import (
    create_relevance_target,
    train_lambdamart_model,
    predict_relevance_scores,
    format_submission_file
)

# --- Configuration ---
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_DIR = "submission"
LOGS_DIR = "logs"

# Development/Debug flag (set to None for full run)
# NROWS_CONFIG = 500000 # For faster development with a subset of data
NROWS_CONFIG = 2000000 # For full run
PERFORM_IMPUTATION = True # Set to False to skip missing value imputation
PERFORM_FEATURE_ENGINEERING = True # Set to False to skip feature engineering


# --- Helper Functions ---

def define_feature_columns(df: pd.DataFrame) -> list[str]:
    """Defines feature columns by excluding IDs, targets, and intermediate columns."""
    excluded_cols = [
        'click_bool', 'booking_bool', 'gross_bookings_usd', 'position',
        'date_time', 'srch_id', 'relevance'
    ]
    potential_features = [col for col in df.columns if col not in excluded_cols]
    # Further filtering can happen based on domain knowledge or feature selection later.
    return potential_features

def preprocess_and_engineer_features(
    df: pd.DataFrame, 
    is_train: bool = True, 
    imputation_params_for_test: dict | None = None
) -> tuple[pd.DataFrame, dict | None]:
    """Handles preprocessing, missing values (conditionally), and feature engineering."""
    print(f"Starting preprocessing & feature engineering for {'training' if is_train else 'test'} data...")
    df_copy = df.copy()
    
    calculated_imputation_params = None # Initialize
    df_processed_stage1 = df_copy # Start with the original copy

    if PERFORM_IMPUTATION:
        print("Performing missing value imputation...")
        df_processed_stage1, calculated_imputation_params = handle_missing_values(
            df_copy, 
            is_train=is_train, 
            imputation_values=imputation_params_for_test
        )
        print(f"Missing values handled. Shape after imputation: {df_processed_stage1.shape}")
    else:
        print("Skipping missing value imputation as per PERFORM_IMPUTATION flag.")
        if not is_train:
            imputation_params_for_test = None
    
    # Apply feature engineering regardless of imputation, on the current state of df_processed_stage1
    if PERFORM_FEATURE_ENGINEERING:
        print("Performing feature engineering...")
        df_engineered = apply_feature_engineering(df_processed_stage1)
    else:
        print("Skipping feature engineering as per PERFORM_FEATURE_ENGINEERING flag.")
        df_engineered = df_processed_stage1 # Pass through if not engineering
    
    print(f"Preprocessing & feature engineering complete. Final shape: {df_engineered.shape}")
    return df_engineered, calculated_imputation_params

def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str, str]:
    """Creates relevance target and defines feature columns."""
    print("Creating relevance target...")
    df_with_target = create_relevance_target(df.copy())
    print("Relevance target created. 'relevance' column added.")
    print(df_with_target['relevance'].value_counts(normalize=True).sort_index())

    feature_cols = define_feature_columns(df_with_target)
    if not feature_cols:
        raise ValueError("No feature columns identified. Check define_feature_columns.")
    
    target_col = 'relevance'
    group_col = 'srch_id'
    print(f"Using {len(feature_cols)} features. First 5: {feature_cols[:5]}")
    return df_with_target, feature_cols, target_col, group_col

def split_data(
    df: pd.DataFrame, 
    target_column: str, 
    group_column: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]: # Corrected return type
    """Splits data into training and validation sets, handling groups."""
    print(f"Splitting data into training ({1-test_size:.0%}) and validation ({test_size:.0%}) sets...")
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    
    train_indices, val_indices = next(gss.split(df, df[target_column], groups=df[group_column]))

    train_set_intermediate = df.iloc[train_indices]
    val_set_intermediate = df.iloc[val_indices]

    train_set = train_set_intermediate.sort_values(by=group_column).reset_index(drop=True)
    val_set = val_set_intermediate.sort_values(by=group_column).reset_index(drop=True)
    
    print(f"Training set shape: {train_set.shape}, Validation set shape: {val_set.shape}")
    return train_set, val_set

def get_categorical_features(
    all_feature_columns: list[str], 
    predefined_categorical_list: list[str]
) -> list[str]:
    """Filters the predefined categorical features to those present in the dataset."""
    final_categorical_features = [col for col in predefined_categorical_list if col in all_feature_columns]
    if len(final_categorical_features) < len(predefined_categorical_list):
        print("Warning: Not all predefined categorical features were found in feature_columns.")
    else:
        print(f"Using the following categorical features: {final_categorical_features}")
    return final_categorical_features

def run_model_training(
    X_train: pd.DataFrame, y_train: pd.Series, group_train_counts: list,
    X_val: pd.DataFrame, y_val: pd.Series, group_val_counts: list,
    feature_columns: list[str], categorical_features: list[str],
    model_params_override: dict | None = None
) -> tuple[lgb.LGBMRanker | None, float | None, dict | None, float | None]:
    """Trains the LambdaMART model and extracts validation NDCG@5 score, params, and duration."""
    print("Training LambdaMART model...")
    
    start_time = time.time()
    model = train_lambdamart_model(
        X_train, y_train, group_train_counts,
        X_val, y_val, group_val_counts,
        feature_names=feature_columns,
        categorical_features=categorical_features,
        params=model_params_override
    )
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Model training complete in {training_duration:.2f} seconds.")

    actual_model_params = model.get_params() if model else None
    ndcg_val_score = None
    if model and hasattr(model, 'best_score_') and model.best_score_:
        try:
            ndcg_val_score = model.best_score_['valid_0']['ndcg@5']
        except KeyError:
            try:
                print("Warning: Could not find 'ndcg@5' in model.best_score_['valid_0']. Trying 'ndcg'.")
                score_candidate = model.best_score_['valid_0']['ndcg']
                ndcg_val_score = score_candidate[0] if isinstance(score_candidate, list) else score_candidate
            except KeyError:
                print("Warning: Could not retrieve NDCG score from model.best_score_.")
    return model, ndcg_val_score, actual_model_params, training_duration

def prepare_and_save_log(
    run_timestamp: str, 
    ndcg_score: float | None, 
    model: lgb.LGBMRanker | None, # Model can be None if training fails
    feature_columns: list[str], 
    log_dir: str,
    model_params: dict | None,         
    training_duration_seconds: float | None, 
    data_summary: dict,                
    categorical_features_used: list[str],
    imputation_params_logged: dict | None # Added to log imputation values
):
    """Prepares log data and saves it to a JSON file."""
    log_data = {
        "run_timestamp": run_timestamp,
        "configuration": {
            "nrows_loaded_train": data_summary.get("nrows_config_train", "all"),
            "nrows_loaded_test": data_summary.get("nrows_config_test", "all")
        },
        "data_summary": {
            "raw_train_shape": str(data_summary.get("raw_train_shape")),
            "final_processed_train_shape": str(data_summary.get("final_processed_train_shape")),
            "train_set_shape": str(data_summary.get("train_set_shape")),
            "val_set_shape": str(data_summary.get("val_set_shape")),
            "X_train_shape": str(data_summary.get("X_train_shape")),
            "y_train_shape": str(data_summary.get("y_train_shape")),
            "num_train_groups": data_summary.get("num_train_groups"),
            "X_val_shape": str(data_summary.get("X_val_shape")),
            "y_val_shape": str(data_summary.get("y_val_shape")),
            "num_val_groups": data_summary.get("num_val_groups"),
            "num_features_used": data_summary.get("num_features"),
            "raw_test_shape": str(data_summary.get("raw_test_shape")),
            "final_processed_test_shape": str(data_summary.get("final_processed_test_shape")),
            "X_test_shape": str(data_summary.get("X_test_shape"))
        },
        "preprocessing_details": {
            "imputation_values_from_train": imputation_params_logged if imputation_params_logged else "Not applicable or only fixed value imputations used"
        },
        "features_used": {
            "all_feature_names": feature_columns,
            "categorical_feature_names": categorical_features_used
        },
        "model_training_results": {
            "parameters_used": model_params if model_params else "Not available",
            "training_duration_seconds": f"{training_duration_seconds:.2f}" if training_duration_seconds is not None else "Not available",
            "validation_ndcg_at_5": ndcg_score if ndcg_score is not None else "Not available"
        },
        "feature_importances_gain": "Not available" 
    }

    if model and hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
        importances = pd.Series(model.feature_importances_, index=feature_columns)
        log_data["feature_importances_gain"] = importances.sort_values(ascending=False).to_dict()
        print("\nFeature Importances (gain) logged.") # Print a confirmation
    else:
        if model: # Only print if model exists but importances are missing
             print("\nCould not retrieve feature importances from the model for logging.")

    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"training_log_{run_timestamp}.json")
    
    try:
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=4, default=str) # default=str handles non-serializable types gracefully
        print(f"Training log saved to {log_file_path}")
    except Exception as e:
        print(f"Error saving log file {log_file_path}: {e}")

def run_prediction_pipeline(
    model: lgb.LGBMRanker, 
    test_file_path: str, 
    feature_columns: list[str], 
    submission_dir: str, 
    run_timestamp: str,
    nrows: int | None = None,
    imputation_params_for_test: dict | None = None
) -> dict: # Return test data summary
    """Handles the prediction pipeline for the test set. Returns test data summary."""
    print("\nStarting prediction pipeline for test set...")
    test_data_log_summary = {
        "raw_test_shape": None,
        "final_processed_test_shape": None,
        "X_test_shape": None
    }
    try:
        test_df = load_data(data_path=test_file_path, nrows=nrows)
    except Exception:
        print(f"Failed to load test data from {test_file_path}. Aborting prediction pipeline.")
        return test_data_log_summary
    
    if test_df is None: # Should be caught by exception now, but good for safety
        return test_data_log_summary 
    test_data_log_summary["raw_test_shape"] = test_df.shape

    test_ids_df = test_df[['srch_id', 'prop_id']].copy()
    
    # Preprocess test data using params from training
    test_df_processed, _ = preprocess_and_engineer_features(
        test_df, 
        is_train=False, 
        imputation_params_for_test=imputation_params_for_test
    )
    test_data_log_summary["final_processed_test_shape"] = test_df_processed.shape

    # Ensure all feature columns (defined from training set) are in the processed test set
    missing_cols_in_test = set(feature_columns) - set(test_df_processed.columns)
    if missing_cols_in_test:
        print(f"Warning: Missing features in test set {missing_cols_in_test}, adding with 0.")
        for col in missing_cols_in_test:
            test_df_processed[col] = 0
    
    # Re-align columns to match training order, just in case, also handles newly added zero columns
    X_test = test_df_processed[feature_columns]
    test_data_log_summary["X_test_shape"] = X_test.shape

    print("Predicting relevance scores on test set...")
    test_predictions = predict_relevance_scores(model, X_test, feature_names=feature_columns)

    submission_df_prep = test_ids_df.copy()
    submission_df_prep['relevance_score'] = test_predictions

    submission_filename = f"submission_{run_timestamp}.csv"
    submission_file_path = os.path.join(submission_dir, submission_filename)
    
    print(f"Formatting submission file and saving to {submission_file_path}...")
    final_submission_df = format_submission_file(
        submission_df_prep,
        group_col='srch_id', item_col='prop_id', score_col='relevance_score'
    )
    
    os.makedirs(os.path.dirname(submission_file_path), exist_ok=True)
    final_submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved: {submission_file_path}")
    return test_data_log_summary


# --- Main Execution ---

def main():
    print("Starting training and prediction pipeline...")
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Load Training Data using load_data from data_loader
    try:
        train_df_raw = load_data(data_path=TRAIN_FILE, nrows=NROWS_CONFIG)
    except Exception:
        print(f"Failed to load training data from {TRAIN_FILE}. Aborting.")
        return

    if train_df_raw is None: # Should be caught by exception now
        return
    raw_train_shape_log = train_df_raw.shape
    
    # 2. Preprocess Training Data, Handle Missing Values, and Engineer Features
    train_df_processed, imputation_values_from_train = preprocess_and_engineer_features(
        train_df_raw, 
        is_train=True
    )
    processed_train_shape_log = train_df_processed.shape
    
    # 3. Create Relevance Target and Define Features (from the processed DataFrame)
    try:
        train_df_final, feature_cols, target_col, group_col = prepare_features_and_target(train_df_processed)
    except ValueError as e:
        print(f"Error during feature/target preparation: {e}")
        return

    # 4. Split Data
    train_set, val_set = split_data(train_df_final, target_col, group_col)
    
    X_train_df = train_set[feature_cols]
    y_train_series = train_set[target_col]
    group_train_counts = train_set.groupby(group_col, sort=False).size().to_numpy()

    X_val_df = val_set[feature_cols]
    y_val_series = val_set[target_col]
    group_val_counts = val_set.groupby(group_col, sort=False).size().to_numpy()

    # 5. Define Categorical Features (from the final list of feature_cols)
    predefined_cats = [
        'site_id', 'visitor_location_country_id', 'prop_country_id',
        'prop_brand_bool', 'promotion_flag', 'srch_destination_id',
        'srch_saturday_night_bool', 'random_bool', 'prop_starrating', 'prop_review_score',
        # Newly added date/time categorical features
        'search_hour_of_day', 'search_day_of_week', 'search_month', 'is_weekend_search'
    ] + [f'comp{i}_{suffix}' for i in range(1, 9) for suffix in ['rate', 'inv']]
    
    categorical_features_final = get_categorical_features(feature_cols, predefined_cats)

    # 6. Train Model
    lgbm_params_to_use = None 
    model, ndcg_val, actual_model_params, model_training_time = run_model_training(
        X_train_df, y_train_series, group_train_counts,
        X_val_df, y_val_series, group_val_counts,
        feature_cols, categorical_features_final,
        model_params_override=lgbm_params_to_use
    )

    training_data_summary = {
        "nrows_config_train": NROWS_CONFIG if NROWS_CONFIG is not None else "all",
        "raw_train_shape": raw_train_shape_log,
        "final_processed_train_shape": processed_train_shape_log, # Shape after all processing
        "train_set_shape": train_set.shape,
        "val_set_shape": val_set.shape,
        "X_train_shape": X_train_df.shape,
        "y_train_shape": y_train_series.shape,
        "num_train_groups": len(group_train_counts),
        "X_val_shape": X_val_df.shape,
        "y_val_shape": y_val_series.shape,
        "num_val_groups": len(group_val_counts),
        "num_features": len(feature_cols)
    }

    print("\n--- Training Summary & Logging ---")
    if ndcg_val is not None:
        print(f"Validation NDCG@5: {ndcg_val:.4f}")
    else:
        print("Validation NDCG@5: Not available or training failed.")

    full_data_summary_for_log = training_data_summary.copy()
    full_data_summary_for_log["nrows_config_test"] = NROWS_CONFIG if NROWS_CONFIG is not None else "all"

    test_summary_info = {}
    if model: 
        test_summary_info = run_prediction_pipeline(
            model, TEST_FILE, feature_cols, SUBMISSION_DIR, run_timestamp, 
            nrows=NROWS_CONFIG, imputation_params_for_test=imputation_values_from_train
        )
        full_data_summary_for_log.update(test_summary_info)
    else:
        print("Skipping prediction pipeline as model training failed or model is None.")

    prepare_and_save_log(
        run_timestamp=run_timestamp, 
        ndcg_score=ndcg_val, 
        model=model, 
        feature_columns=feature_cols, 
        log_dir=LOGS_DIR,
        model_params=actual_model_params,
        training_duration_seconds=model_training_time,
        data_summary=full_data_summary_for_log, 
        categorical_features_used=categorical_features_final,
        imputation_params_logged=imputation_values_from_train # Log the imputation values
    )
    print("--- End Training Summary & Logging ---")

    print("\nFull pipeline (train and predict) finished successfully.")

if __name__ == '__main__':
    main() 