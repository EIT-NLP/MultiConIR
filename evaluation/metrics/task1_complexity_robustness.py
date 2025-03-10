import pandas as pd
from glob import glob
import os

# Define the base directory containing model results
base_dir = 'MultiConIR/datasets/results'

# List of all models to process
models = [
    "bge-en-icl", "bge-reranker-v2-gemma", "bge-reranker-v2-m3", "BM25",
    "e5-mistral", "followIR", "gritlm", "gte-large",
    "gte-Qwen2-1.5B-instruct", "gte-Qwen2-7B-instruct", "jina",
    "llm2vec", "nv-embed"
]

# Define target files for each model
target_files = [
    'Books_Task1_task1_results.csv',
    'Legal Document_Task1_task1_results.csv',
    'Medical Case_Task1_task1_results.csv',
    'Movies_Task1_task1_results.csv',
    'People_Task1_task1_results.csv'
]

results = []

# Iterate over each model directory
for model in models:
    model_dir = os.path.join(base_dir, model)

    if not os.path.exists(model_dir):
        continue  # Skip if model directory does not exist

    all_file_paths = glob(f'{model_dir}/*.csv')

    # Filter files that match target filenames
    filtered_file_paths = [
        file_path for file_path in all_file_paths
        if any(target_file in file_path for target_file in target_files)
    ]

    # Process each filtered file
    for file_path in filtered_file_paths:
        dataset_name = os.path.basename(file_path).split('_hns')[0]  # Extract dataset name

        df = pd.read_csv(file_path)

        model_results = {'Model': model, 'Dataset': dataset_name}

        for i in range(1, 11):
            query_positive_col = f'Query{i}_Positive'
            query_hn_col = f'Query{i}_HN1'

            if query_positive_col in df.columns and query_hn_col in df.columns:
                win_rate = (df[query_positive_col] > df[query_hn_col]).mean()
                average_diff = (df[query_positive_col] - df[query_hn_col]).mean()
            else:
                win_rate = None
                average_diff = None

            # Store results
            model_results[f'Query{i}_WinRate'] = win_rate
            model_results[f'Query{i}_AvgDiff'] = average_diff

        results.append(model_results)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Define column order
ordered_columns = ['Model', 'Dataset']
for i in range(1, 11):
    ordered_columns.append(f'Query{i}_WinRate')
for i in range(1, 11):
    ordered_columns.append(f'Query{i}_AvgDiff')

results_df = results_df[ordered_columns]

# Compute the average row
avg_row = results_df.mean(numeric_only=True).to_dict()
avg_row['Model'] = 'Average'
avg_row['Dataset'] = 'All'

# Append the average row to results
results_df = pd.concat([results_df, pd.DataFrame([avg_row])], ignore_index=True)

# Save results to CSV
output_path = os.path.join(base_dir, "task1_summary.csv")
results_df.to_csv(output_path, index=False)

print(f"Results saved to: {output_path}")
