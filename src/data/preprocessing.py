import pandas as pd

def handle_missing_visitor_hist_starrating(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'visitor_hist_starrating' feature."""
    df['visitor_hist_starrating'] = df['visitor_hist_starrating'].fillna(0)
    return df

def handle_missing_visitor_hist_adr_usd(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'visitor_hist_adr_usd' feature."""
    df['visitor_hist_adr_usd'] = df['visitor_hist_adr_usd'].fillna(0)
    return df

def handle_missing_prop_review_score(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'prop_review_score' feature."""
    df['prop_review_score'] = df['prop_review_score'].fillna(0)
    return df

def handle_missing_prop_location_score2(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'prop_location_score2' feature."""
    median_val = df['prop_location_score2'].median()
    df['prop_location_score2'] = df['prop_location_score2'].fillna(median_val)
    return df

def handle_missing_srch_query_affinity_score(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'srch_query_affinity_score' feature."""
    min_affinity_score = df['srch_query_affinity_score'].min()
    # If all values are NaN, min_affinity_score will be NaN.
    # In that case, or if min_affinity_score is not negative (which is unexpected based on description),
    # we can choose a default small negative value. For now, let's assume it's usually negative.
    # A more robust approach might involve a fixed small negative if min is NaN or non-negative.
    fill_value = min_affinity_score - 100 if pd.notna(min_affinity_score) else -999 # Arbitrary very small value if all are NaN
    df['srch_query_affinity_score'] = df['srch_query_affinity_score'].fillna(fill_value)
    return df

def handle_missing_orig_destination_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'orig_destination_distance' feature."""
    median_val = df['orig_destination_distance'].median()
    df['orig_destination_distance'] = df['orig_destination_distance'].fillna(median_val)
    return df

def handle_missing_comp1_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp1_rate' feature."""
    df['comp1_rate'] = df['comp1_rate'].fillna(0)
    return df

def handle_missing_comp1_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp1_inv' feature."""
    df['comp1_inv'] = df['comp1_inv'].fillna(0)
    return df

def handle_missing_comp1_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp1_rate_percent_diff' feature."""
    df['comp1_rate_percent_diff'] = df['comp1_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp2_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp2_rate' feature."""
    df['comp2_rate'] = df['comp2_rate'].fillna(0)
    return df

def handle_missing_comp2_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp2_inv' feature."""
    df['comp2_inv'] = df['comp2_inv'].fillna(0)
    return df

def handle_missing_comp2_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp2_rate_percent_diff' feature."""
    df['comp2_rate_percent_diff'] = df['comp2_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp3_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp3_rate' feature."""
    df['comp3_rate'] = df['comp3_rate'].fillna(0)
    return df

def handle_missing_comp3_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp3_inv' feature."""
    df['comp3_inv'] = df['comp3_inv'].fillna(0)
    return df

def handle_missing_comp3_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp3_rate_percent_diff' feature."""
    df['comp3_rate_percent_diff'] = df['comp3_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp4_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp4_rate' feature."""
    df['comp4_rate'] = df['comp4_rate'].fillna(0)
    return df

def handle_missing_comp4_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp4_inv' feature."""
    df['comp4_inv'] = df['comp4_inv'].fillna(0)
    return df

def handle_missing_comp4_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp4_rate_percent_diff' feature."""
    df['comp4_rate_percent_diff'] = df['comp4_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp5_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp5_rate' feature."""
    df['comp5_rate'] = df['comp5_rate'].fillna(0)
    return df

def handle_missing_comp5_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp5_inv' feature."""
    df['comp5_inv'] = df['comp5_inv'].fillna(0)
    return df

def handle_missing_comp5_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp5_rate_percent_diff' feature."""
    df['comp5_rate_percent_diff'] = df['comp5_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp6_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp6_rate' feature."""
    df['comp6_rate'] = df['comp6_rate'].fillna(0)
    return df

def handle_missing_comp6_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp6_inv' feature."""
    df['comp6_inv'] = df['comp6_inv'].fillna(0)
    return df

def handle_missing_comp6_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp6_rate_percent_diff' feature."""
    df['comp6_rate_percent_diff'] = df['comp6_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp7_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp7_rate' feature."""
    df['comp7_rate'] = df['comp7_rate'].fillna(0)
    return df

def handle_missing_comp7_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp7_inv' feature."""
    df['comp7_inv'] = df['comp7_inv'].fillna(0)
    return df

def handle_missing_comp7_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp7_rate_percent_diff' feature."""
    df['comp7_rate_percent_diff'] = df['comp7_rate_percent_diff'].fillna(0)
    return df

def handle_missing_comp8_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp8_rate' feature."""
    df['comp8_rate'] = df['comp8_rate'].fillna(0)
    return df

def handle_missing_comp8_inv(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp8_inv' feature."""
    df['comp8_inv'] = df['comp8_inv'].fillna(0)
    return df

def handle_missing_comp8_rate_percent_diff(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'comp8_rate_percent_diff' feature."""
    df['comp8_rate_percent_diff'] = df['comp8_rate_percent_diff'].fillna(0)
    return df

def handle_missing_gross_bookings_usd(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for the 'gross_bookings_usd' feature."""
    df['gross_bookings_usd'] = df['gross_bookings_usd'].fillna(0)
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values for all specified features.
    Each feature is handled by its own function.
    """
    df = df.copy() # Ensure we are working on a copy from the start
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Create indicator for visitor purchase history BEFORE imputing the related columns
    df['has_visitor_purchase_history'] = df['visitor_hist_starrating'].notnull().astype(int)

    df = handle_missing_visitor_hist_starrating(df)
    df = handle_missing_visitor_hist_adr_usd(df)
    df = handle_missing_prop_review_score(df)

    # Create indicator for prop_location_score2 missingness BEFORE imputing
    df['prop_location_score2_is_missing'] = df['prop_location_score2'].isnull().astype(int)

    df = handle_missing_prop_location_score2(df)
    df = handle_missing_srch_query_affinity_score(df)

    # Create indicator for orig_destination_distance missingness BEFORE imputing
    df['orig_destination_distance_is_missing'] = df['orig_destination_distance'].isnull().astype(int)

    df = handle_missing_orig_destination_distance(df)

    # Create indicator variables for competitor data presence BEFORE imputing
    for i in range(1, 9):
        df[f'comp{i}_has_data'] = df[f'comp{i}_rate'].notnull().astype(int)

    df = handle_missing_comp1_rate(df)
    df = handle_missing_comp1_inv(df)
    df = handle_missing_comp1_rate_percent_diff(df)
    df = handle_missing_comp2_rate(df)
    df = handle_missing_comp2_inv(df)
    df = handle_missing_comp2_rate_percent_diff(df)
    df = handle_missing_comp3_rate(df)
    df = handle_missing_comp3_inv(df)
    df = handle_missing_comp3_rate_percent_diff(df)
    df = handle_missing_comp4_rate(df)
    df = handle_missing_comp4_inv(df)
    df = handle_missing_comp4_rate_percent_diff(df)
    df = handle_missing_comp5_rate(df)
    df = handle_missing_comp5_inv(df)
    df = handle_missing_comp5_rate_percent_diff(df)
    df = handle_missing_comp6_rate(df)
    df = handle_missing_comp6_inv(df)
    df = handle_missing_comp6_rate_percent_diff(df)
    df = handle_missing_comp7_rate(df)
    df = handle_missing_comp7_inv(df)
    df = handle_missing_comp7_rate_percent_diff(df)
    df = handle_missing_comp8_rate(df)
    df = handle_missing_comp8_inv(df)
    df = handle_missing_comp8_rate_percent_diff(df)
    # df = handle_missing_gross_bookings_usd(df) # Not included in test set
    
    return df
