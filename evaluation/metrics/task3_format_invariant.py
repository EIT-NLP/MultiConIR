import pandas as pd
import os

# Define base directory containing results
base_dir = 'MultiConIR/datasets/results'

# List of models to process
models = [
    "bge-en-icl", "bge-reranker-v2-gemma", "bge-reranker-v2-m3", "BM25",
    "e5-mistral", "followIR", "gritlm", "gte-large",
    "gte-Qwen2-1.5B-instruct", "gte-Qwen2-7B-instruct", "jina",
    "llm2vec", "nv-embed"
]

# Define instruction-style and descriptive-style dataset filenames
instruction_files = {
    'scotus': 'Legal Document_Task2_&_3_task2_results.csv',
    'books': 'Books_Task2_&_3_task2_results.csv',
    'movies': 'Movies_Task2_&_3_task2_results.csv',
    'medical': 'Medical Case_Task2_&_3_task2_results.csv',
    'people': 'People_Task2_&_3_task2_results.csv'
}

descriptive_files = {
    'scotus': 'Legal Document_Task2_&_3_task3_results.csv',
    'books': 'Books_Task2_&_3_task3_results.csv',
    'movies': 'Movies_Task2_&_3_task3_results.csv',
    'medical': 'Medical Case_Task2_&_3_task3_results.csv',
    'people': 'People_Task2_&_3_task3_results.csv'
}

# Function to load CSV data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return None

# Function to compute win rate change between instruction-style and descriptive-style queries
def compute_change_rate(inst_df, desc_df):
    query_col = 'Query10_Positive'
    hn_cols = [f'Query10_HN{i}' for i in range(1, 11)]

    total_comparisons = 0
    changed_comparisons = 0

    for col in [query_col] + hn_cols[:-1]:  # Compare Positive vs HN1, HN1 vs HN2, ..., HN9 vs HN10
        next_col = hn_cols[hn_cols.index(col) + 1] if col != query_col else hn_cols[0]

        if col in inst_df.columns and next_col in inst_df.columns and col in desc_df.columns and next_col in desc_df.columns:
            inst_comparison = inst_df[col] > inst_df[next_col]
            desc_comparison = desc_df[col] > desc_df[next_col]

            # Count the number of times ranking changes
            changes = (inst_comparison != desc_comparison).sum()
            changed_comparisons += changes
            total_comparisons += len(inst_df)

    return changed_comparisons / total_comparisons if total_comparisons > 0 else 0

# Compute change rates for all models and datasets
change_results = []

for model in models:
    model_dir = os.path.join(base_dir, model)

    if not os.path.exists(model_dir):
        continue  # Skip if model directory does not exist

    for dataset in instruction_files.keys():
        inst_path = os.path.join(model_dir, instruction_files[dataset])
        desc_path = os.path.join(model_dir, descriptive_files[dataset])

        inst_df = load_data(inst_path)
        desc_df = load_data(desc_path)

        if inst_df is not None and desc_df is not None:
            change_rate = compute_change_rate(inst_df, desc_df)
            change_results.append({
                'Model': model,
                'Dataset': dataset,
                'Change Rate': change_rate
            })

# Convert results to DataFrame and save to CSV
change_results_df = pd.DataFrame(change_results)
output_file = os.path.join(base_dir, 'query_style_change_results.csv')
change_results_df.to_csv(output_file, index=False)

print("Win rate change rate statistics completed. Results saved to:", output_file)
print(change_results_df)