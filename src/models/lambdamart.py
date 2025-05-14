import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit

def create_relevance_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates the relevance target variable based on booking_bool and click_bool.
    Relevance: 5 (booked), 1 (clicked), 0 (neither).
    """
    # Ensure 'booking_bool' and 'click_bool' are present
    if 'booking_bool' not in df.columns or 'click_bool' not in df.columns:
        raise ValueError("DataFrame must contain 'booking_bool' and 'click_bool' columns.")

    df['relevance'] = 0
    df.loc[df['click_bool'] == 1, 'relevance'] = 1
    df.loc[df['booking_bool'] == 1, 'relevance'] = 5 # Overwrites click_bool = 1 if also booked
    return df

def train_lambdamart_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    group_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    group_val: pd.Series,
    feature_names: list[str],
    categorical_features: list[str] | None = None,
    params: dict | None = None
) -> lgb.LGBMRanker:
    """
    Initializes and trains an LGBMRanker model.
    """
    if params is None:
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5], # Evaluate NDCG@5 as per competition
            'boosting_type': 'gbdt',
            'n_estimators': 100, # Placeholder, can be tuned
            'learning_rate': 0.05, # Placeholder, can be tuned
            'num_leaves': 31, # Placeholder, can be tuned
            'random_state': 42,
            'n_jobs': -1,
            'importance_type': 'gain',
        }

    ranker = lgb.LGBMRanker(**params)

    print("Training LightGBM Ranker...")
    ranker.fit(
        X_train[feature_names],
        y_train,
        group=group_train,
        eval_set=[(X_val[feature_names], y_val)],
        eval_group=[group_val],
        eval_at=[5], # Evaluate NDCG@5 during training
        callbacks=[lgb.early_stopping(10, verbose=True)],
        categorical_feature=categorical_features
    )
    print("Training complete.")
    return ranker

def predict_relevance_scores(
    model: lgb.LGBMRanker,
    X_test: pd.DataFrame,
    feature_names: list[str]
) -> pd.Series:
    """
    Predicts relevance scores for the test set.
    """
    print("Predicting relevance scores...")
    predictions = model.predict(X_test[feature_names])
    print("Prediction complete.")
    return pd.Series(predictions, index=X_test.index, name='relevance_score')

def format_submission_file(
    test_df_with_scores: pd.DataFrame,
    group_col: str = 'srch_id',
    item_col: str = 'prop_id',
    score_col: str = 'relevance_score'
) -> pd.DataFrame:
    """
    Formats the predictions into the Kaggle submission format.
    Sorts items within each group by score and selects top N (though Kaggle expects all sorted).
    """
    print("Formatting submission file...")
    # Sort by group (srch_id) and then by score (descending)
    submission_df = test_df_with_scores.sort_values(
        by=[group_col, score_col],
        ascending=[True, False]
    )
    # Select only the required columns
    submission_df = submission_df[[group_col, item_col]]
    submission_df = submission_df.rename(columns={group_col: 'SearchId', item_col: 'PropertyId'})
    print("Submission formatting complete.")
    return submission_df

# Example usage (will be part of a main script or notebook later)
if __name__ == '__main__':
    # This section is for demonstration and will be refined.
    # It assumes data loading and preprocessing have been done.

    print("Starting LambdaMART example workflow...")
    # 1. Load preprocessed data (mocking this for now)
    #    In a real scenario, you would load from a file, e.g., using pd.read_csv
    #    and ensure 'src/data/preprocessing.py' has been run.
    print("Simulating data loading and preprocessing...")
    sample_size = 100000 # Using a smaller sample for quick demo
    data = {
        'srch_id': [i // 5 for i in range(sample_size)], # 5 items per search query
        'prop_id': [i % 1000 for i in range(sample_size)], # Some property IDs
        'feature1': [i * 0.1 for i in range(sample_size)],
        'feature2': [i * -0.05 + 10 for i in range(sample_size)],
        'booking_bool': [1 if i % 20 == 0 else 0 for i in range(sample_size)], # ~5% booking rate
        'click_bool': [1 if i % 5 == 0 else 0 for i in range(sample_size)],   # ~20% click rate (includes bookings)
        # Add other features as per your dataset
    }
    all_df = pd.DataFrame(data)

    # Ensure prop_id is unique within each srch_id for this mock data, or adjust if necessary
    all_df = all_df.drop_duplicates(subset=['srch_id', 'prop_id']).reset_index(drop=True)


    # 2. Create relevance target
    all_df = create_relevance_target(all_df.copy()) # Use .copy() to avoid SettingWithCopyWarning

    # 3. Define features and target
    #    These would be all your preprocessed columns except IDs, target, and intermediate booleans
    feature_columns = [col for col in all_df.columns if col.startswith('feature')]
    # In a real scenario, feature_columns would be more extensive:
    # e.g., ['prop_starrating', 'prop_review_score', 'price_usd', 'visitor_hist_starrating', ...]
    target_column = 'relevance'
    group_column = 'srch_id'

    if not feature_columns:
        print("Warning: No feature columns found starting with 'feature'. Using dummy features.")
        all_df['dummy_feature_for_demo'] = range(len(all_df))
        feature_columns = ['dummy_feature_for_demo']


    # 4. Split data into train and validation (maintaining group integrity)
    #    For a real split, ensure validation set reflects test set characteristics if possible.
    print("Splitting data into training and validation sets...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # We need to pass groups to the split method.
    # Ensure all_df is sorted by group_column if GroupShuffleSplit has issues with unsorted groups.
    # However, GroupShuffleSplit should handle it.
    train_idx, val_idx = next(gss.split(all_df, all_df[target_column], groups=all_df[group_column]))

    train_df = all_df.iloc[train_idx].copy() # Use .copy() to avoid future SettingWithCopyWarning
    validation_df = all_df.iloc[val_idx].copy() # Use .copy()

    # Crucial step: DataFrames must be sorted by group_column before creating group arrays for LightGBM
    train_df = train_df.sort_values(by=group_column).reset_index(drop=True)
    validation_df = validation_df.sort_values(by=group_column).reset_index(drop=True)
    
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    group_train = train_df.groupby(group_column, sort=False).size().to_numpy()

    X_val = validation_df[feature_columns]
    y_val = validation_df[target_column]
    group_val = validation_df.groupby(group_column, sort=False).size().to_numpy()


    # 5. Train model
    print(f"Feature names for training: {feature_columns}")
    lgbm_ranker = train_lambdamart_model(
        X_train, y_train, group_train,
        X_val, y_val, group_val,
        feature_names=feature_columns,
        categorical_features=['promotion_flag']
    )

    # 6. Predict on a "test" set (using validation set for this example)
    #    In a real scenario, load and preprocess actual test_df.
    #    The test_df_for_prediction should be sorted by group_column if you want to align scores easily,
    #    though predict function itself doesn't require it. format_submission_file will sort it.
    print("Simulating prediction on a test set (using validation set as proxy)...")
    
    test_df_for_prediction = validation_df.copy() # Or load your actual test data
    
    # The X_test for prediction must contain only feature columns in the correct order
    X_test_features_only = test_df_for_prediction[feature_columns]

    predicted_scores = predict_relevance_scores(
        lgbm_ranker,
        X_test_features_only,
        feature_names=feature_columns
    )
    test_df_for_prediction['relevance_score'] = predicted_scores
    

    # 7. Format for submission
    submission = format_submission_file(
        test_df_for_prediction, # This df must have srch_id, prop_id, and relevance_score
        group_col='srch_id',
        item_col='prop_id',
        score_col='relevance_score'
    )
    print("\nSample of formatted submission file:")
    print(submission.head())

    # You would then save this submission DataFrame to a CSV file
    # submission.to_csv('submission.csv', index=False)
    # print("\nSubmission file 'submission.csv' would be generated.")

    print("\nLambdaMART example workflow finished.") 