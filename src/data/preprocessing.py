import pandas as pd

def handle_missing_prop_review_score(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'prop_review_score' feature."""
    df['prop_review_score'] = df['prop_review_score'].fillna(0)
    return df

def handle_missing_srch_query_affinity_score(df: pd.DataFrame, min_val_override: float | None = None) -> pd.DataFrame:
    """Handles missing values for the 'srch_query_affinity_score' feature."""
    if min_val_override is not None:
        fill_value = min_val_override - 100 # Consistent with original logic if min_val is from train
    else:
        min_affinity_score_df = df['srch_query_affinity_score'].min()
        fill_value = min_affinity_score_df - 100 if pd.notna(min_affinity_score_df) else -999
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(fill_value)
    return df


def handle_missing_values(df: pd.DataFrame, is_train: bool = True, imputation_values: dict | None = None) -> tuple[pd.DataFrame, dict | None]:
    """
    Handles missing values for all specified features.
    Each feature is handled by its own function.
    If is_train is True, calculates imputation statistics (medians, etc.) and returns them.
    If is_train is False, uses pre-calculated imputation_values.
    """
    df = df.copy()
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])

    calculated_imputation_values = {}

    # prop_review_score
    if 'prop_review_score' in df.columns:
        df = handle_missing_prop_review_score(df)

    # srch_query_affinity_score
    if 'srch_query_affinity_score' in df.columns:
        query_affinity_min = None
        if is_train:
            query_affinity_min = df['srch_query_affinity_score'].min()
            calculated_imputation_values['srch_query_affinity_score_min'] = query_affinity_min
        elif imputation_values:
            query_affinity_min = imputation_values.get('srch_query_affinity_score_min')
        df = handle_missing_srch_query_affinity_score(df, min_val_override=query_affinity_min)
    
    return df, (calculated_imputation_values if is_train else None)
