

import pandas as pd
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_directory = 'MultiConIR/datasets/results/gte/'

from tqdm import tqdm 



def gte_large_task1(data_path, max_length):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task1_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    model_path = 'Alibaba-NLP/gte-large-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/code/.cache/huggingface")
    model = AutoModel.from_pretrained(model_path, cache_dir="/code/.cache/huggingface", trust_remote_code=True).to(device)

    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        other_positives = []
        for i in range(7):
            other_index = (index + i + 1) % len(df) 
            other_positives.append(df.loc[other_index, 'Positive'])

        row_result = {"Index": index}

        for i in range(1, 11): 
            Query_col = f'Query{i}'
            Positive_col = f'Positive'
            HN_col = f'HN{i}'
            HN2_col = f'HN_{i}'

            input_texts = [row[Query_col], row[Positive_col], row[HN_col], row[HN2_col]] + other_positives

            batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = F.normalize(embeddings, p=2, dim=1)

            scores = (embeddings[:1] @ embeddings[1:].T) * 100
            scores_list = scores[0].tolist()
            print(f"Scores for Query{i}: {scores_list}")

            row_result[f"Query{i}_Positive"] = scores_list[0]
            row_result[f"Query{i}_HN1"] = scores_list[1]
            row_result[f"Query{i}_HN2"] = scores_list[2]
            for j in range(7):
                row_result[f"Query{i}_EN{j+1}"] = scores_list[3 + j]

        all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")


def gte_large_task2(data_path):

    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task2_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    model_path = 'Alibaba-NLP/gte-large-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/code/.cache/huggingface")
    model = AutoModel.from_pretrained(model_path, cache_dir="/code/.cache/huggingface", trust_remote_code=True).to(device)

    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        input_texts = [row['Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)

        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        scores_list = scores[0].tolist()
        print(f"Scores for row {index + 1}: {scores_list}")

        row_result = {
            "Index": index,
            "Query10_Positive": scores_list[0]
        }
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores_list[i]

        all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")


def gte_large_task3(data_path):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task3_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    model_path = 'Alibaba-NLP/gte-large-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/code/.cache/huggingface")
    model = AutoModel.from_pretrained(model_path, cache_dir="/code/.cache/huggingface", trust_remote_code=True).to(device)

    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        input_texts = [row['Natural_Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        batch_dict = tokenizer(input_texts, max_length=1024, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]
            embeddings = F.normalize(embeddings, p=2, dim=1)

        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        scores_list = scores[0].tolist()
        print(f"Scores for row {index + 1}: {scores_list}")

        row_result = {
            "Index": index,
            "Query10_Positive": scores_list[0]
        }
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores_list[i]

        all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def gte_large_task4(data_path):

    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task4_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    model_path = 'Alibaba-NLP/gte-large-en-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/code/.cache/huggingface")
    model = AutoModel.from_pretrained(model_path, cache_dir="/code/.cache/huggingface", trust_remote_code=True).to(device)

    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        input_texts = [row['Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        batch_dict = tokenizer(input_texts, max_length=2048, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        # 生成嵌入
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = outputs.last_hidden_state[:, 0]

            embeddings = F.normalize(embeddings, p=2, dim=1)

        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        scores_list = scores[0].tolist()
        print(f"Scores for row {index + 1}: {scores_list}")

        row_result = {
            "Index": index,
            "Query10_Positive": scores_list[0]
        }
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores_list[i]

        all_results.append(row_result)

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



    task1_data_paths = [
        '/code/luxuan/datasets/dataset_v2/books/v2_books_hns.csv',
        '/code/luxuan/datasets/dataset_v2/legal/v2_legal_hns.csv',
        '/code/luxuan/datasets/dataset_v2/medical case/v2_medical case_hns.csv',
        '/code/luxuan/datasets/dataset_v2/movies/v2_movies_hns.csv',
        '/code/luxuan/datasets/dataset_v2/people/v2_people_hns.csv'
    ]


    for data_path in task1_data_paths:
        gte_large_task1(data_path)

    for data_path in task2_data_paths:
        gte_large_task2(data_path)

    for data_path in task3_data_paths:
        gte_large_task3(data_path)

    for data_path in task4_data_paths:
        gte_large_task4(data_path)
        