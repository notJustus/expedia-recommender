import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import sys
import os

# Add src directory to Python path to allow direct imports
# This assumes run_training.py is in the src directory
# and sibling directories like 'data' and 'models' are also in 'src'
sys.path.append(os.path.dirname(__file__))

# Corrected import paths assuming preprocessing.py is in src/data/
# and lambdamart.py is in src/models/
from data.preprocessing import handle_missing_values
from models.lambdamart import (
    create_relevance_target,
    train_lambdamart_model,
    # predict_relevance_scores, # Not using for now, but good to have
    # format_submission_file # Not using for now
)

def define_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Defines the list of feature columns to be used for training.
    Excludes ID columns, raw target columns, and the final relevance score.
    """
    excluded_cols = [
        'srch_id', 'date_time', 'site_id', 'visitor_location_country_id',
        'prop_country_id', 'prop_id', 'srch_destination_id',
        'click_bool', 'booking_bool', 'gross_bookings_usd', 'position', # Target-related or leaky
        'relevance' # Target variable
    ]
    # Features created during preprocessing (e.g., _is_missing, _has_data) are included.
    feature_cols = [col for col in df.columns if col not in excluded_cols]
    return feature_cols

def main():
    print("Starting training pipeline...")

    # 1. Load data
    # The path to train.csv should be relative to the WORKSPACE ROOT, not this script's location.
    # If train.csv is in <workspace_root>/data/train.csv
    data_file_path = 'data/train.csv'
    print(f"Loading {data_file_path}...")
    try:
        # Load a subset for faster development/testing if needed
        # For full run, set nrows=None or remove it.
        train_df = pd.read_csv(data_file_path, nrows=500000) # Using 500k rows for quicker dev
    except FileNotFoundError:
        print(f"Error: {data_file_path} not found. Please ensure the path is correct from the workspace root.")
        return
    print(f"Data loaded. Shape: {train_df.shape}")

    # 2. Preprocess data
    print("Preprocessing data...")
    train_df = handle_missing_values(train_df.copy()) # Use .copy()
    print(f"Preprocessing complete. Shape after preprocessing: {train_df.shape}")

    # 3. Create relevance target
    print("Creating relevance target...")
    train_df = create_relevance_target(train_df.copy()) # Use .copy()
    print("Relevance target created. 'relevance' column added.")
    print(train_df['relevance'].value_counts(normalize=True).sort_index())


    # 4. Define features and target
    feature_columns = define_feature_columns(train_df)
    if not feature_columns:
        print("Error: No feature columns were identified. Please check define_feature_columns.")
        return

    target_column = 'relevance'
    group_column = 'srch_id'
    print(f"Using {len(feature_columns)} features. First 5: {feature_columns[:5]}")


    # 5. Split data into training and validation sets
    print("Splitting data into training (80%) and validation (20%) sets...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    # Data should be sorted by group_column for creating group arrays correctly for LightGBM.
    # The split itself is fine on unsorted data, but X_train, y_train, group_train (and for val)
    # must correspond to data sorted by group_id.
    
    # Perform the split on the original (or preprocessed) dataframe
    train_indices, val_indices = next(gss.split(
        train_df, 
        train_df[target_column], 
        groups=train_df[group_column]
    ))

    train_set_intermediate = train_df.iloc[train_indices]
    val_set_intermediate = train_df.iloc[val_indices]

    # Now, sort these subsets by group_column before extracting X, y, and group counts
    train_set = train_set_intermediate.sort_values(by=group_column).reset_index(drop=True)
    val_set = val_set_intermediate.sort_values(by=group_column).reset_index(drop=True)
    
    X_train = train_set[feature_columns]
    y_train = train_set[target_column]
    group_train = train_set.groupby(group_column, sort=False).size().to_numpy()

    X_val = val_set[feature_columns]
    y_val = val_set[target_column]
    group_val = val_set.groupby(group_column, sort=False).size().to_numpy()

    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}")
    print(f"Number of groups in training set: {len(group_train)}, total items: {group_train.sum()}")
    print(f"Number of groups in validation set: {len(group_val)}, total items: {group_val.sum()}")


    # 6. Train LambdaMART model
    print("Training LambdaMART model...")
    model = train_lambdamart_model(
        X_train, y_train, group_train,
        X_val, y_val, group_val,
        feature_names=feature_columns
        # Pass custom params if needed, e.g., params={'n_estimators': 50}
    )
    print("Model training complete.")

    # 7. Show feature importances
    if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
        print("\nFeature Importances (gain):")
        importances = pd.Series(model.feature_importances_, index=feature_columns)
        print(importances.sort_values(ascending=False).head(20)) # Print top 20
    else:
        print("\nCould not retrieve feature importances from the model.")

    print("\nTraining pipeline finished successfully.")

if __name__ == '__main__':
    main() 