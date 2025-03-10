import os
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from transformers import AutoConfig
target_directory = 'MultiConIR/datasets/results/nv-embed-v2/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/code/.cache/huggingface/models--nvidia--NV-Embed-v2"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

query_prefix = "Given a document search query, retrieve relevant documents that match the query\nQuery: "
passage_prefix = ""



def compute_embeddings(queries, passages, max_length=6144, batch_size=4):

    print("Computing query embeddings...")
    query_embeddings = model.encode(queries, instruction=query_prefix, max_length=512)
    # query_embeddings = query_embeddings.to(device)
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    print("Computing passage embeddings in batches...")
    passage_embeddings_list = []
    for i in range(0, len(passages), batch_size):
        batch_passages = passages[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(passages) + batch_size - 1) // batch_size}")
        
        batch_embeddings = model.encode(batch_passages, instruction=passage_prefix, max_length=max_length)
        batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

        passage_embeddings_list.append(batch_embeddings)

        torch.cuda.empty_cache()

    passage_embeddings = torch.cat(passage_embeddings_list, dim=0)
    
    return query_embeddings, passage_embeddings

def nv_embed_task1(data_path, max_length=6144):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task1_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
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

            with torch.no_grad():
                embeddings1, embeddings2 = compute_embeddings(input_texts[:1], input_texts[1:], max_length)

            scores = (embeddings1 @ embeddings2.T) * 100
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

def nv_embed_task2(data_path, max_length=6144):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task2_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        input_texts = [row['Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        print(len(input_texts))
        # 生成嵌入
        with torch.no_grad():
            embeddings1, embeddings2 = compute_embeddings(input_texts[:1], input_texts[1:], max_length)

            torch.cuda.empty_cache()

        scores = (embeddings1 @ embeddings2.T) * 100
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

def nv_embed_task3(data_path, max_length=6144):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task3_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        input_texts = [row['Natural_Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        print(len(input_texts))
        # 生成嵌入
        with torch.no_grad():
            embeddings1, embeddings2 = compute_embeddings(input_texts[:1], input_texts[1:], max_length)

            torch.cuda.empty_cache()

        scores = (embeddings1 @ embeddings2.T) * 100
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

def nv_embed_task4(data_path, max_length=2048):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task4_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        input_texts = [row['Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        print(len(input_texts))
        with torch.no_grad():
            embeddings1, embeddings2 = compute_embeddings(input_texts[:1], input_texts[1:], max_length)

            torch.cuda.empty_cache()

        scores = (embeddings1 @ embeddings2.T) * 100
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


    max_length = 2048

    for data_path in task1_data_paths:
        nv_embed_task1(data_path, max_length=max_length)

    for data_path in task2_data_paths:
        nv_embed_task2(data_path, max_length=max_length)

    for data_path in task3_data_paths:
        nv_embed_task3(data_path, max_length=max_length)

    for data_path in task4_data_paths:
        nv_embed_task4(data_path, max_length=max_length)