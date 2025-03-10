import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_directory = 'MultiConIR/datasets/results/gte-Qwen2-1.5B-instruct/'

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def gte_qwen_task1(data_path, max_length):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task1_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    model_path = '/code/.cache/huggingface/gte-Qwen2-1.5B-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

    all_results = []

    # Iterate through each row to calculate similarity
    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        # Retrieve other seven positive samples
        other_positives = []
        for i in range(7):
            other_index = (index + i + 1) % len(df)  # Ensure valid row indexing
            other_positives.append(df.loc[other_index, 'Positive'])

        # Store current row's results
        row_result = {"Index": index}

        for i in range(1, 11):  # Iterate through Query1 to Query10
            Query_col = f'Query{i}'
            Positive_col = f'Positive'
            HN_col = f'HN{i}'
            HN2_col = f'HN_{i}'

            # Combine input texts: Query, Positive, HN, HN2, and other positives
            input_texts = [row[Query_col], row[Positive_col], row[HN_col], row[HN2_col]] + other_positives

            # Tokenize inputs and move to GPU
            batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {key: value.to(device) for key, value in batch_dict.items()}

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)

            # Calculate similarity scores
            scores = (embeddings[:1] @ embeddings[1:].T) * 100
            scores_list = scores[0].tolist()
            print(f"Scores for Query{i}: {scores_list}")

            # Save similarity results to independent columns
            row_result[f"Query{i}_Positive"] = scores_list[0]
            row_result[f"Query{i}_HN1"] = scores_list[1]
            row_result[f"Query{i}_HN2"] = scores_list[2]
            for j in range(7):
                row_result[f"Query{i}_EN{j+1}"] = scores_list[3 + j]

        # Append current row's results to overall results
        all_results.append(row_result)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")


def gte_qwen_task2(data_path):
    # Read the CSV file
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task2_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # Load the model and tokenizer
    model_path = '/code/.cache/huggingface/gte-Qwen2-1.5B-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

    # Store computation results
    all_results = []

    # Iterate through each row to calculate similarity
    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        # Combine Query10, Positive, and HN1 to HN10 as input texts
        input_texts = [row['Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        # Tokenize inputs and create a batch
        batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {key: value.to(device) for key, value in batch_dict.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Calculate similarity scores
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        scores_list = scores[0].tolist()
        print(f"Scores for row {index + 1}: {scores_list}")

        # Save similarity results as independent columns
        row_result = {
            "Index": index,
            "Query10_Positive": scores_list[0]
        }
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores_list[i]

        # Append current row's results to overall results
        all_results.append(row_result)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def gte_qwen_task3(data_path):
    # Read the CSV file
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task3_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # Load the model and tokenizer
    model_path = '/code/.cache/huggingface/gte-Qwen2-1.5B-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

    # Store computation results
    all_results = []

    # Iterate through each row to calculate similarity
    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        # Combine Query10, Positive, and HN1 to HN10 as input texts
        input_texts = [row['Natural_Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        # Tokenize inputs and create a batch
        batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {key: value.to(device) for key, value in batch_dict.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Calculate similarity scores
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        scores_list = scores[0].tolist()
        print(f"Scores for row {index + 1}: {scores_list}")

        # Save similarity results as independent columns
        row_result = {
            "Index": index,
            "Query10_Positive": scores_list[0]
        }
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores_list[i]

        # Append current row's results to overall results
        all_results.append(row_result)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")



def gte_qwen_task4(data_path):
    # Read the CSV file
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task4_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    # Load the model and tokenizer
    model_path = '/code/.cache/huggingface/gte-Qwen2-1.5B-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=False).to(device)

    # Store computation results
    all_results = []

    # Iterate through each row to calculate similarity
    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        # Combine Query10, Positive, and HN1 to HN10 as input texts
        input_texts = [row['Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        # Tokenize inputs and create a batch
        batch_dict = tokenizer(input_texts, max_length=1024, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {key: value.to(device) for key, value in batch_dict.items()}

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Calculate similarity scores
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        scores_list = scores[0].tolist()
        print(f"Scores for row {index + 1}: {scores_list}")

        # Save similarity results as independent columns
        row_result = {
            "Index": index,
            "Query10_Positive": scores_list[0]
        }
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores_list[i]

        # Append current row's results to overall results
        all_results.append(row_result)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

if __name__ == '__main__':

    task1_data_paths = ['MultiConIR/datasets/Task1/Books_Task1.csv',
                    'MultiConIR/datasets/Task1/Legal Document_Task1.csv',
                    'MultiConIR/datasets/Task1/Medical Case_Task1.csv',
                    'MultiConIR/datasets/Task1/Movies_Task1.csv',
                    'MultiConIR/datasets/Task1/People_Task1.csv'

    ]


    task2_data_paths = ['MultiConIR/datasets/Task2_&_3/Books_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Legal Document_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Medical Case_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Movies_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/People_Task2_&_3.csv'

    ]


    task3_data_paths = ['MultiConIR/datasets/Task2_&_3/Books_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Legal Document_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Medical Case_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Movies_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/People_Task2_&_3.csv'

    ]

    task4_data_paths = [
        'MultiConIR/datasets/Task4/Books_512.csv',
        'MultiConIR/datasets/Task4/Books_1024.csv',

        'MultiConIR/datasets/Task4/Legal Document_512.csv',
        'MultiConIR/datasets/Task4/Legal Document_1024.csv',

        'MultiConIR/datasets/Task4/Medical Case_512.csv',
        'MultiConIR/datasets/Task4/Medical Case_1024.csv',

        'MultiConIR/datasets/Task4/Movies_512.csv',
        'MultiConIR/datasets/Task4/Movies_1024.csv',

        'MultiConIR/datasets/Task4/People_512.csv',
        'MultiConIR/datasets/Task4/People_1024.csv'
    ]


    for data_path in task1_data_paths:
        gte_qwen_task1(data_path)

    for data_path in task2_data_paths:
        gte_qwen_task2(data_path)

    for data_path in task3_data_paths:
        gte_qwen_task3(data_path)

    for data_path in task4_data_paths:
        gte_qwen_task4(data_path)
        




