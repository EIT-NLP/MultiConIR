import pandas as pd
from glob import glob
import os

# Base directory where model results are stored
base_dir = 'MultiConIR/datasets/results'

# List of models to process
models = [
    "bge-en-icl", "bge-reranker-v2-gemma", "bge-reranker-v2-m3", "BM25",
    "e5-mistral", "followIR", "gritlm", "gte-large",
    "gte-Qwen2-1.5B-instruct", "gte-Qwen2-7B-instruct", "jina",
    "llm2vec", "nv-embed"
]

# Target filenames
target_files = [
    'Books_Task2_&_3_task2_results.csv',
    'Legal Document_Task2_&_3_task2_results.csv',
    'Medical Case_Task2_&_3_task2_results.csv',
    'Movies_Task2_&_3_task2_results.csv',
    'People_Task2_&_3_task2_results.csv'
]

# Initialize results list
results = []

# Iterate over all model directories
for model in models:
    model_dir = os.path.join(base_dir, model)

    if not os.path.exists(model_dir):
        continue  # Skip if model directory does not exist

    all_file_paths = glob(f'{model_dir}/*.csv')

    # Filter target files
    filtered_file_paths = [
        file_path for file_path in all_file_paths
        if any(target_file in file_path for target_file in target_files)
    ]

    # Process each filtered file
    for file_path in filtered_file_paths:
        dataset_name = os.path.basename(file_path).split('_task2')[0]  # Extract dataset name

        df = pd.read_csv(file_path)

        query_col = 'Query10_Positive'
        hn_cols = [f'Query10_HN{i}' for i in range(1, 11)]

        model_results = {'Model': model, 'Dataset': dataset_name}

        win_rates = []

        # Compare Positive vs HN1
        if query_col in df.columns and hn_cols[0] in df.columns:
            win_rate_positive_vs_hn1 = (df[query_col] > df[hn_cols[0]]).mean()
        else:
            win_rate_positive_vs_hn1 = None
        win_rates.append(win_rate_positive_vs_hn1)
        model_results['Positive_vs_HN1'] = win_rate_positive_vs_hn1

        # Compare HN1 vs HN2, ..., HN9 vs HN10
        for i in range(len(hn_cols) - 1):
            hn1_col = hn_cols[i]
            hn2_col = hn_cols[i + 1]

            if hn1_col in df.columns and hn2_col in df.columns:
                win_rate = (df[hn1_col] > df[hn2_col]).mean()
            else:
                win_rate = None

            win_rates.append(win_rate)
            model_results[f'HN{i+1}_vs_HN{i+2}'] = win_rate

        # Append results
        results.append(model_results)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Compute the average row
avg_row = results_df.mean(numeric_only=True).to_dict()
avg_row['Model'] = 'Average'
avg_row['Dataset'] = 'All'

# Append the average row
results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)

# Save results to CSV
output_path = os.path.join(base_dir, "task2_results_summary.csv")
results_df.to_csv(output_path, index=False)

print(f"Task2 results have been saved to: {output_path}")