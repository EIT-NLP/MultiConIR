from llm2vec import LLM2Vec
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from huggingface_hub import login
import os
import faiss
from torch.nn import functional as F
from typing import List
from sklearn.metrics import ndcg_score, precision_score
import numpy as np

target_directory = 'MultiConIR/datasets/results/llm2vec/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def llm2vec_task1(data_path, l2v, batch_size=8, max_length=1024):

    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task1_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size]
        print(f'Processing rows {start + 1} to {start + len(batch_df)} / {len(df)}')

        other_positives = []
        for i in range(7):
            other_index = (start + i + 1) % len(df)
            other_positives.append(df.loc[other_index, 'Positive'])

        for idx, row in batch_df.iterrows():
            row_result = {"Index": row.name}

            for i in range(1, 11): 
                Query_col = f'Query{i}'
                Positive_col = f'Positive'
                HN1_col = f'HN{i}'
                HN2_col = f'HN_{i}'

                input_texts = [row[Query_col], row[Positive_col], row[HN1_col], row[HN2_col]] + other_positives

                with torch.no_grad():
                    embeddings = l2v.encode(input_texts)
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                query_embedding = embeddings[0].unsqueeze(0)  # [1, D]
                doc_embeddings = embeddings[1:]  # [N, D]
                scores = torch.mm(query_embedding, doc_embeddings.T) * 100
                scores_list = scores[0].tolist()

                row_result[f"Query{i}_Positive"] = scores_list[0]
                row_result[f"Query{i}_HN1"] = scores_list[1]
                row_result[f"Query{i}_HN2"] = scores_list[2]
                for j in range(7):
                    row_result[f"Query{i}_EN{j+1}"] = scores_list[3 + j]

            all_results.append(row_result)

            results_df = pd.DataFrame(all_results)
            results_df.to_csv(new_path, index=False)
            print(f"Results saved to {new_path}")


def llm2vec_task2(data_path, l2v, batch_size=8, max_length=8192):

    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task2_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    instruction = "Given a document search query, retrieve relevant documents that match the query."

    all_results = []
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size]
        print(f'Processing rows {start + 1} to {start + len(batch_df)} / {len(df)}')

        query_inputs = [[instruction, row['Query10']] for _, row in batch_df.iterrows()]
        doc_inputs = []
        for _, row in batch_df.iterrows():
            doc_inputs.append(row['Positive'])
            doc_inputs.extend([row[f'HN{i}'] for i in range(1, 11)])

        with torch.no_grad():
            query_embeddings = l2v.encode(query_inputs)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

            doc_embeddings = l2v.encode(doc_inputs)
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

        for idx, row in enumerate(batch_df.iterrows()):
            row_index, row_data = row

            query_embedding = query_embeddings[idx].unsqueeze(0)  # [1, D]
            start_idx = idx * 11
            end_idx = start_idx + 11
            doc_subset_embeddings = doc_embeddings[start_idx:end_idx]  # [11, D]

            scores = torch.mm(query_embedding, doc_subset_embeddings.T) * 100
            scores_list = scores[0].tolist()

            row_result = {
                "Index": row_index,
                "Query10_Positive": scores_list[0],
            }
            for i in range(1, 11):
                row_result[f"Query10_HN{i}"] = scores_list[i]

            all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def llm2vec_task3(data_path, l2v, batch_size=8, max_length=8192):

    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task3_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    instruction = "Given a document search query, retrieve relevant documents that match the query."

    all_results = []
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size]
        print(f'Processing rows {start + 1} to {start + len(batch_df)} / {len(df)}')

        query_inputs = [[instruction, row['Natural_Query10']] for _, row in batch_df.iterrows()]
        doc_inputs = []
        for _, row in batch_df.iterrows():
            doc_inputs.append(row['Positive'])
            doc_inputs.extend([row[f'HN{i}'] for i in range(1, 11)])

        with torch.no_grad():
            query_embeddings = l2v.encode(query_inputs)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

            doc_embeddings = l2v.encode(doc_inputs)
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

        for idx, row in enumerate(batch_df.iterrows()):
            row_index, row_data = row

            query_embedding = query_embeddings[idx].unsqueeze(0)  # [1, D]
            start_idx = idx * 11
            end_idx = start_idx + 11
            doc_subset_embeddings = doc_embeddings[start_idx:end_idx]  # [11, D]

            scores = torch.mm(query_embedding, doc_subset_embeddings.T) * 100
            scores_list = scores[0].tolist()

            row_result = {
                "Index": row_index,
                "Query10_Positive": scores_list[0],
            }
            for i in range(1, 11):
                row_result[f"Query10_HN{i}"] = scores_list[i]

            all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def llm2vec_task4(data_path, l2v, batch_size=8, max_length=2048):

    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task4_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    instruction = "Given a document search query, retrieve relevant documents that match the query."

    all_results = []

    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start:start + batch_size]
        print(f'Processing rows {start + 1} to {start + len(batch_df)} / {len(df)}')
        query_inputs = [[instruction, row['Query10']] for _, row in batch_df.iterrows()]
        doc_inputs = []
        for _, row in batch_df.iterrows():
            doc_inputs.append(row['Positive'])
            doc_inputs.extend([row[f'HN{i}'] for i in range(1, 11)])

        with torch.no_grad():
            query_embeddings = l2v.encode(query_inputs)
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

            doc_embeddings = l2v.encode(doc_inputs)
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

        for idx, row in enumerate(batch_df.iterrows()):
            row_index, row_data = row

            query_embedding = query_embeddings[idx].unsqueeze(0)  # [1, D]
            start_idx = idx * 11
            end_idx = start_idx + 11
            doc_subset_embeddings = doc_embeddings[start_idx:end_idx]  # [11, D]

            scores = torch.mm(query_embedding, doc_subset_embeddings.T) * 100
            scores_list = scores[0].tolist()

            row_result = {
                "Index": row_index,
                "Query10_Positive": scores_list[0],
            }
            for i in range(1, 11):
                row_result[f"Query10_HN{i}"] = scores_list[i]

            all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")



if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
    config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True)
    base_model = AutoModel.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
    ).to(device)
    peft_model = PeftModel.from_pretrained(
        base_model,
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    ).to(device)
    peft_model = peft_model.merge_and_unload()
    l2v = LLM2Vec(peft_model, tokenizer, pooling_mode="mean")

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



max_length = 2048

for data_path in task1_data_paths.items():
    llm2vec_task1(data_path, l2v, max_length=max_length)

for data_path, max_length in task2_data_paths.items():
    llm2vec_task2(data_path, l2v, max_length=max_length)

for data_path, max_length in task3_data_paths.items():
    llm2vec_task3(data_path, l2v, max_length=max_length)

for data_path, max_length in task4_data_paths.items():
    llm2vec_task4(data_path, l2v, max_length=max_length)