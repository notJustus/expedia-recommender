import os
import json
import pandas as pd
from typing import List, Dict, Any

LOGS_DIR = "logs"
TOP_N_FEATURES = 10 # Number of top features to display for the best run

def load_all_log_data(logs_dir: str) -> List[Dict[str, Any]]:
    """Loads all JSON log files from the specified directory."""
    all_log_data = []
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory '{logs_dir}' not found.")
        return all_log_data

    for filename in os.listdir(logs_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(logs_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    log_content = json.load(f)
                    all_log_data.append(log_content)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from file: {file_path}")
            except Exception as e:
                print(f"Warning: Error reading file {file_path}: {e}")
    return all_log_data

def extract_relevant_metrics(log_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extracts key metrics from loaded log data into a pandas DataFrame."""
    extracted_runs = []
    for log in log_data:
        run_timestamp = log.get("run_timestamp", "N/A")
        
        config = log.get("configuration", {})
        nrows_train = config.get("nrows_loaded_train", "N/A")

        model_results = log.get("model_training_results", {})
        ndcg_score_raw = model_results.get("validation_ndcg_at_5", "N/A")
        training_duration = model_results.get("training_duration_seconds", "N/A")
        
        data_summary = log.get("data_summary", {})
        num_features = data_summary.get("num_features_used", "N/A")

        # feature_importances = log.get("feature_importances_gain", {}) # Not directly used in this df

        ndcg_score = pd.NA
        if isinstance(ndcg_score_raw, (int, float)):
            ndcg_score = float(ndcg_score_raw)
        elif isinstance(ndcg_score_raw, str) and ndcg_score_raw.lower() not in ["not available", "n/a"]:
            try:
                ndcg_score = float(ndcg_score_raw)
            except ValueError:
                ndcg_score = pd.NA 

        extracted_runs.append({
            "Timestamp": run_timestamp,
            "NDCG@5 (Val)": ndcg_score,
            "Training Time (s)": training_duration,
            "Num Features": num_features,
            "NRows Train Cfg": nrows_train,
            "Full Log": log 
        })
    
    if not extracted_runs:
        return pd.DataFrame()
        
    df = pd.DataFrame(extracted_runs)
    for col in ["Training Time (s)", "Num Features"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.sort_values(by="NDCG@5 (Val)", ascending=False, na_position='last')

def print_executive_summary(all_runs_df: pd.DataFrame):
    """Prints an executive summary of the training runs."""
    if all_runs_df.empty:
        print("No log data found or processed.")
        return

    print("\n--- Executive Summary of Training Runs ---")
    print(f"Total runs analyzed: {len(all_runs_df)}")

    valid_ndcg_runs = all_runs_df.dropna(subset=["NDCG@5 (Val)"])

    if valid_ndcg_runs.empty:
        print("No runs found with valid NDCG@5 scores.")
    else:
        best_run_details = valid_ndcg_runs.iloc[0]
        print("\nBest Performing Run:")
        print(f"  Timestamp: {best_run_details['Timestamp']}")
        print(f"  Validation NDCG@5: {best_run_details['NDCG@5 (Val)']:.4f}")
        print(f"  Training Time: {best_run_details['Training Time (s)']:.2f}s" if pd.notna(best_run_details['Training Time (s)']) else "  Training Time: N/A")
        print(f"  Number of Features: {best_run_details['Num Features']}")
        print(f"  NRows Config (Train): {best_run_details['NRows Train Cfg']}")

        best_run_full_log = best_run_details["Full Log"]
        feature_importances = best_run_full_log.get("feature_importances_gain")
        if isinstance(feature_importances, dict) and feature_importances and feature_importances != "Not available":
            print(f"\n  Top {TOP_N_FEATURES} Feature Importances (Gain) for Best Run:")
            sorted_features = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:TOP_N_FEATURES]):
                print(f"    {i+1}. {feature}: {float(importance):.4f}") # Ensure importance is float for formatting
        else:
            print("\n  Feature importances not available or not in expected dictionary format for the best run.")
            
        model_params = best_run_full_log.get("model_training_results", {}).get("parameters_used", "Not available")
        if model_params != "Not available" and isinstance(model_params, dict):
            print("\n  Model Parameters for Best Run:")
            for param, value in model_params.items():
                print(f"    {param}: {value}")
        else:
            print("\n  Model parameters not available for the best run.")

    print("\n--- Summary of All Runs (Sorted by NDCG@5 Desc) ---")
    display_cols = ["Timestamp", "NDCG@5 (Val)", "Training Time (s)", "Num Features", "NRows Train Cfg"]
    formatted_df = all_runs_df[display_cols].copy()
    if "NDCG@5 (Val)" in formatted_df:
         formatted_df["NDCG@5 (Val)"] = formatted_df["NDCG@5 (Val)"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    if "Training Time (s)" in formatted_df:
        formatted_df["Training Time (s)"] = formatted_df["Training Time (s)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

    print(formatted_df.to_string(index=False))
    print("-----------------------------------------\n")

def main():
    print(f"Analyzing logs from directory: {LOGS_DIR}")
    raw_log_data = load_all_log_data(LOGS_DIR)
    
    if not raw_log_data:
        print("No log files found or loaded. Exiting.")
        return
        
    processed_df = extract_relevant_metrics(raw_log_data)
    print_executive_summary(processed_df)

if __name__ == "__main__":
    main() 