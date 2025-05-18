import optuna
import pandas as pd
import sys
import os
import datetime
import json
import time

# Add src directory to Python path
sys.path.append(os.path.dirname(__file__)) # If run from src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # If run from root project directory

from data.data_loader import load_data
from data.preprocessing import handle_missing_values # Ensure this function is suitable or adjust
from data.feature_engineering import apply_feature_engineering
from models.lambdamart import (
    create_relevance_target,
    train_lambdamart_model,
    # predict_relevance_scores, # Not needed for tuning typically
    # format_submission_file # Not needed for tuning
)
# Import functions from run_training.py that can be reused
# We might need to refactor run_training.py to make some parts more easily callable
# For now, let's assume we can reuse:
# - define_feature_columns (if we make it standalone)
# - preprocess_and_engineer_features (already in run_training.py, good)
# - prepare_features_and_target (already in run_training.py, good)
# - split_data (already in run_training.py, good)
# - get_categorical_features (already in run_training.py, good)
# - run_model_training (already in run_training.py, perfect for optimization)

# It's often better to copy and adapt parts of run_training.py logic here
# to avoid direct dependency or to simplify.
# For now, we'll re-define/adapt the necessary helper functions from run_training.py

# --- Configuration ---
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
# TEST_FILE = os.path.join(DATA_DIR, "test.csv") # Not needed for tuning
LOGS_DIR = "logs/hyperparameter_tuning" # Separate log dir for tuning

# Development/Debug flag for tuning (use a smaller subset for faster trials)
NROWS_TUNING = 5000  # Adjust as needed for speed vs. representativeness
PERFORM_IMPUTATION_TUNING = True
PERFORM_FEATURE_ENGINEERING_TUNING = True

# Optuna settings
N_TRIALS = 5 # Number of Optuna trials
STUDY_NAME_PREFIX = "lambdamart_tuning"

# --- Helper Functions (adapted or copied from run_training.py) ---

def define_feature_columns(df: pd.DataFrame) -> list[str]:
    """Defines feature columns by excluding IDs, targets, and intermediate columns."""
    excluded_cols = [
        'click_bool', 'booking_bool', 'gross_bookings_usd', 'position',
        'date_time', 'srch_id', 'relevance'
    ]
    # Add any other columns that are side-products of feature engineering but not features themselves
    # e.g. if some intermediate columns are created and then used to make final features.
    # This list might need adjustment based on exact features created.
    potential_features = [col for col in df.columns if col not in excluded_cols]
    return potential_features

# We'll use preprocess_and_engineer_features from run_training.py directly if possible,
# or adapt it here if changes are needed for tuning.
# For now, assuming we can call the one from run_training's imports:
from run_training import (
    preprocess_and_engineer_features,
    prepare_features_and_target,
    split_data,
    get_categorical_features,
    run_model_training # This is the core function we'll use
)


# --- Optuna Objective Function ---
def objective(trial: optuna.Trial, train_df_processed_full: pd.DataFrame) -> float:
    """
    Optuna objective function to train and evaluate the LambdaMART model.
    """
    print(f"--- Starting Trial {trial.number} ---")
    start_trial_time = time.time()

    # 1. Define Hyperparameters to Tune
    # These are common LightGBM parameters for ranking. Adjust ranges as needed.
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5], # Evaluate NDCG@5
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0), # Row subsampling
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0), # Feature subsampling
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True), # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True), # L2 regularization
        # "label_gain": [0, 1, 5], # Align with relevance scores if not default
        "random_state": 42, # For reproducibility within a trial
        "n_jobs": -1,       # Use all available cores
        "verbose": -1, # Suppress LightGBM's own verbosity during Optuna trials
    }

    # 2. Prepare Data (using the pre-loaded and processed data)
    # We pass train_df_processed_full to avoid reloading and reprocessing for each trial
    print("Preparing features and target for this trial...")
    try:
        train_df_final, feature_cols, target_col, group_col = prepare_features_and_target(train_df_processed_full.copy())
    except ValueError as e:
        print(f"Error during feature/target preparation in trial: {e}")
        # Optuna handles exceptions by pruning the trial or marking it as failed
        raise optuna.exceptions.TrialPruned(f"Data preparation failed: {e}")


    print("Splitting data for this trial...")
    # Use a fixed random state for splitting for fair comparison across trials using the same dataset
    train_set, val_set = split_data(train_df_final, target_col, group_col, random_state=123)

    X_train_df = train_set[feature_cols]
    y_train_series = train_set[target_col]
    group_train_counts = train_set.groupby(group_col, sort=False).size().to_numpy()

    X_val_df = val_set[feature_cols]
    y_val_series = val_set[target_col]
    group_val_counts = val_set.groupby(group_col, sort=False).size().to_numpy()

    # Define Categorical Features
    predefined_cats = [
        'site_id', 'visitor_location_country_id', 'prop_country_id',
        'prop_brand_bool', 'promotion_flag', 'srch_destination_id',
        'srch_saturday_night_bool', 'random_bool', 'prop_starrating', 'prop_review_score',
        'search_hour_of_day', 'search_day_of_week', 'search_month', 'is_weekend_search'
    ] + [f'comp{i}_{suffix}' for i in range(1, 9) for suffix in ['rate', 'inv']]
    categorical_features_final = get_categorical_features(feature_cols, predefined_cats)

    # 3. Train Model using run_model_training
    print(f"Training model with params: {params}")
    model, ndcg_val, actual_model_params, model_training_time = run_model_training(
        X_train_df, y_train_series, group_train_counts,
        X_val_df, y_val_series, group_val_counts,
        feature_cols, categorical_features_final,
        model_params_override=params # Pass the suggested params
    )

    end_trial_time = time.time()
    trial_duration = end_trial_time - start_trial_time
    print(f"--- Trial {trial.number} Finished. Duration: {trial_duration:.2f}s. Validation NDCG@5: {ndcg_val} ---")

    if ndcg_val is None: # Handle cases where training might fail or score isn't returned
        return 0.0 # Or raise optuna.exceptions.TrialPruned()

    return ndcg_val


# --- Main Execution for Hyperparameter Tuning ---
def main_tuning():
    print("Starting hyperparameter tuning pipeline...")
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"{STUDY_NAME_PREFIX}_{run_timestamp}"

    # 1. Load and Preprocess Data ONCE before starting trials
    print(f"Loading initial training data (nrows={NROWS_TUNING})...")
    try:
        train_df_raw = load_data(data_path=TRAIN_FILE, nrows=NROWS_TUNING)
    except Exception as e:
        print(f"Failed to load training data from {TRAIN_FILE}. Aborting tuning. Error: {e}")
        return

    if train_df_raw is None:
        return

    print("Preprocessing and engineering features for the tuning dataset...")
    # For tuning, we typically don't need to save imputation_values, as we're not applying to a test set here.
    # The preprocess_and_engineer_features from run_training.py handles this.
    train_df_processed, _ = preprocess_and_engineer_features(
        train_df_raw,
        is_train=True # Ensures correct handling if imputation values are calculated
    )
    # `train_df_processed` will be passed to the objective function.

    # 2. Create and Run Optuna Study
    # We'll use a lambda to pass the preprocessed data to the objective function
    objective_with_data = lambda trial: objective(trial, train_df_processed_full=train_df_processed)
    
    # Store results in a SQLite database for persistence (optional but recommended)
    # storage_name = f"sqlite:///{LOGS_DIR}/{study_name}.db"
    # os.makedirs(LOGS_DIR, exist_ok=True) # Ensure log directory exists
    # print(f"Optuna study storage: {storage_name}")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        # storage=storage_name, # Uncomment to use SQLite storage
        # load_if_exists=True   # Uncomment to resume study if it exists
    )

    print(f"Starting Optuna optimization with {N_TRIALS} trials...")
    study.optimize(objective_with_data, n_trials=N_TRIALS, timeout=None) # Set timeout in seconds if needed

    # 3. Log Results
    print("--- Hyperparameter Tuning Complete ---")
    print(f"Study Name: {study.study_name}")
    print(f"Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"Best trial (Number {best_trial.number}):")
    print(f"  Value (NDCG@5): {best_trial.value:.6f}")
    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save best parameters to a JSON file
    best_params_file = os.path.join(LOGS_DIR, f"best_params_{study_name}.json")
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(best_params_file, 'w') as f:
        json.dump(best_trial.params, f, indent=4)
    print(f"Best parameters saved to: {best_params_file}")

    # You can also save the full study for later analysis
    # For example, using joblib:
    # import joblib
    # study_file = os.path.join(LOGS_DIR, f"study_{study_name}.pkl")
    # joblib.dump(study, study_file)
    # print(f"Full Optuna study saved to: {study_file}")

    print("--- End Hyperparameter Tuning ---")

if __name__ == '__main__':
    main_tuning() 