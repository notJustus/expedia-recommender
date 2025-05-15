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
    predict_relevance_scores, # Kept for potential future use, but not called for test
    format_submission_file # Kept for potential future use, but not called
)

# --- Configuration ---

DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
# TEST_FILE = os.path.join(DATA_DIR, "test.csv") # Test file loading skipped
SUBMISSION_DIR = "submission" # Kept for structure, but not used
LOGS_DIR = "logs"

# Development/Debug flag (set to None for full run)
# NROWS_CONFIG = 500000 # For faster development with a subset of data
NROWS_CONFIG = 2000000 # For full run on training data
PERFORM_IMPUTATION = True # Set to False to skip missing value imputation
PERFORM_FEATURE_ENGINEERING = True # Set to False to skip feature engineering


# --- Helper Functions (Identical to run_training.py) ---

def define_feature_columns(df: pd.DataFrame) -> list[str]:
    """Defines feature columns by excluding IDs, targets, and intermediate columns."""
    excluded_cols = [
        'click_bool', 'booking_bool', 'gross_bookings_usd', 'position',
        'date_time', 'srch_id', 'relevance'
    ]
    # Also exclude any new aggregate columns that might be created and are not raw features
    # Example: 'is_family_search', 'is_recurring_customer'
    # This list might need to be dynamically updated based on feature_engineering.py
    
    # A more robust way might be to explicitly list known non-features
    # or tag features during their creation.
    # For now, we rely on the initial exclusion list.
    
    potential_features = [col for col in df.columns if col not in excluded_cols]
    
    # Filter out columns that were created for specific analyses if they are not actual features
    # (e.g., 'is_family_search' if it was only for a specific groupby in EDA)
    # This part is tricky without knowing exactly which columns from apply_feature_engineering are final features.
    # Assuming apply_feature_engineering returns a DataFrame where additional columns are valid features.
    
    return potential_features

def preprocess_and_engineer_features(
    df: pd.DataFrame, 
    is_train: bool = True, 
    imputation_params_for_test: dict | None = None # Not used in this script for test data
) -> tuple[pd.DataFrame, dict | None]:
    """Handles preprocessing, missing values (conditionally), and feature engineering."""
    print(f"Starting preprocessing & feature engineering for {'training' if is_train else 'test (skipped)'} data...")
    df_copy = df.copy()
    
    calculated_imputation_params = None # Initialize
    df_processed_stage1 = df_copy 

    if PERFORM_IMPUTATION:
        print("Performing missing value imputation...")
        # For training-only, we always calculate imputation params
        df_processed_stage1, calculated_imputation_params = handle_missing_values(
            df_copy, 
            is_train=True, # Always True for this script's context
            imputation_values=None # We calculate them from training data
        )
        print(f"Missing values handled. Shape after imputation: {df_processed_stage1.shape}")
    else:
        print("Skipping missing value imputation as per PERFORM_IMPUTATION flag.")
        calculated_imputation_params = None 
            
    if PERFORM_FEATURE_ENGINEERING:
        print("Performing feature engineering...")
        df_engineered = apply_feature_engineering(df_processed_stage1)
    else:
        print("Skipping feature engineering as per PERFORM_FEATURE_ENGINEERING flag.")
        df_engineered = df_processed_stage1
    
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
) -> tuple[pd.DataFrame, pd.DataFrame]: 
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
    # This list should be maintained based on actual categorical features created
    default_cats = [
        'site_id', 'visitor_location_country_id', 'prop_country_id', 
        'prop_id', 'prop_brand_bool', 'srch_destination_id',
        'srch_saturday_night_bool', 
        # Add any new categorical features from feature_engineering.py
        'is_recurring_customer', # Example if it's treated as categorical
        'search_hour_of_day', 'search_day_of_week', 'search_month', 'is_weekend_search' 
    ]
    # Ensure only features actually present in the dataframe are passed
    final_categorical_features = [col for col in default_cats if col in all_feature_columns]
    
    if not final_categorical_features and predefined_categorical_list:
         print("Warning: No predefined categorical features were found in the final feature columns. Using an empty list.")
    elif len(final_categorical_features) < len(predefined_categorical_list):
        print(f"Warning: Some predefined categorical features were not found. Using: {final_categorical_features}")
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
    if model and hasattr(model, 'best_score_') and model.best_score_ and 'valid_0' in model.best_score_:
        # Try to get 'ndcg@5' first, then 'ndcg'
        if 'ndcg@5' in model.best_score_['valid_0']:
            ndcg_val_score = model.best_score_['valid_0']['ndcg@5']
        elif 'ndcg' in model.best_score_['valid_0']:
            score_candidate = model.best_score_['valid_0']['ndcg']
            # Handle cases where ndcg might be a list (e.g., if multiple metrics use 'ndcg')
            ndcg_val_score = score_candidate[0] if isinstance(score_candidate, list) else score_candidate
            print(f"Warning: 'ndcg@5' not found, using 'ndcg' score: {ndcg_val_score}")
        else:
            print("Warning: Neither 'ndcg@5' nor 'ndcg' found in model.best_score_['valid_0'].")
    else:
        print("Warning: Could not retrieve NDCG score from model.best_score_ (model or best_score_ might be missing).")
        
    return model, ndcg_val_score, actual_model_params, training_duration


def prepare_and_save_log(
    run_timestamp: str, 
    ndcg_score: float | None, 
    model: lgb.LGBMRanker | None, 
    feature_columns: list[str], 
    log_dir: str,
    model_params: dict | None,         
    training_duration_seconds: float | None, 
    data_summary: dict,                
    categorical_features_used: list[str],
    imputation_params_logged: dict | None 
):
    """Prepares log data and saves it to a JSON file."""
    log_data = {
        "run_timestamp": run_timestamp,
        "run_type": "training_only", # Added to distinguish log type
        "configuration": {
            "nrows_loaded_train": data_summary.get("nrows_config_train", "all"),
            # "nrows_loaded_test": data_summary.get("nrows_config_test", "all") # Test data not loaded
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
            # Test data shapes removed
        },
        "model_performance": {
            "validation_ndcg_at_5": f"{ndcg_score:.6f}" if ndcg_score is not None else None,
            "training_duration_seconds": f"{training_duration_seconds:.2f}" if training_duration_seconds is not None else None,
        },
        "model_details": {
            "model_parameters": model_params if model_params else (model.get_params() if model else None),
            "num_features_used": len(feature_columns),
            "feature_columns_used_sample": feature_columns[:20], # Log a sample
            "categorical_features_used": categorical_features_used,
        },
        "imputation_details": imputation_params_logged if imputation_params_logged else "Not performed or no params generated"
    }

    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"training_log_{run_timestamp}.json")
    
    try:
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=4)
        print(f"Training log saved to {log_file_path}")
    except Exception as e:
        print(f"Error saving training log: {e}")

# --- Prediction Pipeline (Kept for reference, but not called in main) ---
def run_prediction_pipeline(
    model: lgb.LGBMRanker, 
    test_file_path: str, 
    feature_columns: list[str], 
    categorical_features: list[str], # Added categorical features
    submission_dir: str, 
    run_timestamp: str,
    nrows: int | None = None,
    imputation_params_for_test: dict | None = None
) -> dict: 
    """Loads test data, preprocesses, predicts, and formats submission file."""
    print(f"\n--- Starting Prediction Pipeline for Test Data ---")
    print(f"Loading test data from {test_file_path}...")
    test_df_raw = load_data(test_file_path, nrows=nrows)
    if test_df_raw.empty:
        print("Test data is empty. Skipping prediction.")
        return {"raw_test_shape": (0,0), "final_processed_test_shape": (0,0)}

    raw_test_shape = test_df_raw.shape
    print(f"Raw test data shape: {raw_test_shape}")

    # Preprocess test data (is_train=False) using imputation_params from training
    test_df_processed, _ = preprocess_and_engineer_features(
        test_df_raw, 
        is_train=False, 
        imputation_params_for_test=imputation_params_for_test
    )
    final_processed_test_shape = test_df_processed.shape
    print(f"Processed test data shape: {final_processed_test_shape}")

    # Ensure all feature columns are present, fill missing ones with a default (e.g., 0 or NaN)
    # This can happen if some features were data-dependent in training (e.g. rare categories)
    for col in feature_columns:
        if col not in test_df_processed.columns:
            print(f"Warning: Feature '{col}' not found in processed test data. Filling with 0.")
            test_df_processed[col] = 0 
            
    X_test = test_df_processed[feature_columns]
    
    print("Predicting relevance scores on test data...")
    test_df_processed['predicted_relevance'] = predict_relevance_scores(model, X_test)
    
    print("Formatting submission file...")
    submission_df = format_submission_file(test_df_processed)
    
    os.makedirs(submission_dir, exist_ok=True)
    submission_file_path = os.path.join(submission_dir, f"submission_{run_timestamp}.csv")
    submission_df.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to {submission_file_path}")
    
    return {
        "raw_test_shape": raw_test_shape, 
        "final_processed_test_shape": final_processed_test_shape,
        "submission_file_path": submission_file_path
    }

# --- Main Execution ---
def main():
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"--- Starting Model Training Run: {run_timestamp} ---")

    overall_start_time = time.time()
    data_summary_log = {"nrows_config_train": NROWS_CONFIG if NROWS_CONFIG is not None else "all"}

    # 1. Load Training Data
    print(f"Loading training data from {TRAIN_FILE}...")
    train_df_raw = load_data(TRAIN_FILE, nrows=NROWS_CONFIG)
    if train_df_raw.empty:
        print("Training data is empty. Exiting.")
        return
    data_summary_log["raw_train_shape"] = train_df_raw.shape
    print(f"Raw training data shape: {train_df_raw.shape}")

    # 2. Preprocess and Engineer Features for Training Data
    # This also returns imputation_params calculated from training data
    train_df_processed, imputation_params = preprocess_and_engineer_features(
        train_df_raw, 
        is_train=True
    )
    data_summary_log["final_processed_train_shape"] = train_df_processed.shape
    
    # 3. Prepare Features and Target for Training
    train_df_final, feature_columns, target_col, group_col = prepare_features_and_target(train_df_processed)
    
    # 4. Split Training Data into Train/Validation
    train_set, val_set = split_data(train_df_final, target_col, group_col)
    data_summary_log["train_set_shape"] = train_set.shape
    data_summary_log["val_set_shape"] = val_set.shape

    # Prepare data for LightGBM
    X_train = train_set[feature_columns]
    y_train = train_set[target_col]
    group_train_counts = train_set.groupby(group_col).size().tolist()
    
    X_val = val_set[feature_columns]
    y_val = val_set[target_col]
    group_val_counts = val_set.groupby(group_col).size().tolist()

    data_summary_log.update({
        "X_train_shape": X_train.shape, "y_train_shape": y_train.shape, "num_train_groups": len(group_train_counts),
        "X_val_shape": X_val.shape, "y_val_shape": y_val.shape, "num_val_groups": len(group_val_counts)
    })

    # Define categorical features
    # This list should be comprehensive of all categorical features created
    # predefined_categorical_list = [
    #     'site_id', 'visitor_location_country_id', 'prop_country_id', 
    #     'prop_id', 'prop_brand_bool', 'srch_destination_id',
    #     'srch_saturday_night_bool', 
    #     # Add other known categorical features that might be generated
    #     'is_recurring_customer', 'is_domestic_travel',
    #     'search_hour_of_day', 'search_day_of_week', 'search_month', 'is_weekend_search'
    # ]
    predefined_categorical_list = [
        'site_id', 'visitor_location_country_id', 'prop_country_id',
        'prop_brand_bool', 'promotion_flag', 'srch_destination_id',
        'srch_saturday_night_bool', 'random_bool', 'prop_starrating', 'prop_review_score',
        # Newly added date/time categorical features
        'search_hour_of_day', 'search_day_of_week', 'search_month', 'is_weekend_search'
    ] + [f'comp{i}_{suffix}' for i in range(1, 9) for suffix in ['rate', 'inv']]
    categorical_features_to_use = get_categorical_features(feature_columns, predefined_categorical_list)

    # 5. Run Model Training
    model, ndcg_val, final_model_params, training_duration = run_model_training(
        X_train, y_train, group_train_counts,
        X_val, y_val, group_val_counts,
        feature_columns, categorical_features_to_use,
        model_params_override=None # Pass specific params here if needed
    )
    
    if model is None:
        print("Model training failed. Skipping further steps.")
    else:
        print(f"Achieved Validation NDCG@5: {ndcg_val if ndcg_val is not None else 'N/A'}")

    # 6. Prepare and Save Log (Training information only)
    prepare_and_save_log(
        run_timestamp=run_timestamp,
        ndcg_score=ndcg_val,
        model=model,
        feature_columns=feature_columns,
        log_dir=LOGS_DIR,
        model_params=final_model_params,
        training_duration_seconds=training_duration,
        data_summary=data_summary_log,
        categorical_features_used=categorical_features_to_use,
        imputation_params_logged=imputation_params
    )

    # Prediction pipeline on test data is SKIPPED in this script

    overall_end_time = time.time()
    print(f"--- Training-Only Run Completed in {(overall_end_time - overall_start_time):.2f} seconds ---")

if __name__ == "__main__":
    main() 