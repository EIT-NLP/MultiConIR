import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
torch.cuda.empty_cache()

from gritlm import GritLM
from scipy.spatial.distance import cosine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GritLM("/code/.cache/huggingface/GritLM-7B", torch_dtype="auto").to(device)
target_directory = 'MultiConIR/datasets/results/gritlm/'


def gritlm_instruction(task_type):
    if task_type == "similarity_computation":
        return "<|user|>\nGiven a document search query, retrieve relevant documents that match the query\n<|embed|>\n"
    elif task_type == "embedding_computation":
        return "<|user|>\nGiven a document search query, retrieve relevant documents that match the query\n<|embed|>\n"
    else:
        return "<|embed|>\n"  


def compute_similarity(model, input_texts, task_type="similarity_computation",max_length=6144):
    batch_dict = model.encode(input_texts, instruction=gritlm_instruction(task_type),max_length=max_length) 
    scores = []
    for i in range(1, len(input_texts)):
        cosine_sim = 1 - cosine(batch_dict[0], batch_dict[i])  
        scores.append(cosine_sim * 100)  
    return scores

def gritlm_task1(data_path):
    df = pd.read_csv(data_path)
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
            Positive_col = 'Positive'
            HN_col = f'HN{i}'
            HN2_col = f'HN_{i}'

            input_texts = [row[Query_col], row[Positive_col], row[HN_col], row[HN2_col]] + other_positives

            batch_dict = model.encode(input_texts, instruction=gritlm_instruction("similarity_computation"),max_length=6144)

            scores = []
            for j in range(1, len(input_texts)):
                cosine_sim = 1 - cosine(batch_dict[0], batch_dict[j])  
                scores.append(cosine_sim * 100)

            row_result[f"Query{i}_Positive"] = scores[0]  
            row_result[f"Query{i}_HN1"] = scores[1]  
            row_result[f"Query{i}_HN2"] = scores[2]  
            for j in range(7):
                row_result[f"Query{i}_EN{j+1}"] = scores[3 + j]  

        all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    file_name = os.path.basename(data_path).replace(".csv", "_task1_results.csv")
    new_path = os.path.join(target_directory, file_name)

    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")
    

def gritlm_task2(data_path):
    df = pd.read_csv(data_path)
    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        input_texts = [row['Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        batch_dict = model.encode(input_texts, instruction=gritlm_instruction("similarity_computation"),max_length=6144) 

        scores = []
        for i in range(1, len(input_texts)):
            cosine_sim = 1 - cosine(batch_dict[0], batch_dict[i]) 
            scores.append(cosine_sim * 100)

        row_result = {"Index": index, "Query10_Positive": scores[0]}  
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores[i]  

        all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    file_name = os.path.basename(data_path).replace(".csv", "_task2_results.csv")
    new_path = os.path.join(target_directory, file_name)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")


def gritlm_task3(data_path):
    df = pd.read_csv(data_path)
    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        input_texts = [row['Natural_Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        batch_dict = model.encode(input_texts, instruction=gritlm_instruction("similarity_computation"),max_length=1024 )

        scores = []
        for i in range(1, len(input_texts)):
            cosine_sim = 1 - cosine(batch_dict[0], batch_dict[i])  
            scores.append(cosine_sim * 100)

        row_result = {"Index": index, "Query10_Positive": scores[0]} 
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores[i]  

        all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    file_name = os.path.basename(data_path).replace(".csv", "_task3_results.csv")
    new_path = os.path.join(target_directory, file_name)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")


def gritlm_task4(data_path):
    df = pd.read_csv(data_path)
    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        input_texts = [row['Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        batch_dict = model.encode(input_texts, instruction=gritlm_instruction("similarity_computation"),max_length=1024)

        scores = []
        for i in range(1, len(input_texts)):
            cosine_sim = 1 - cosine(batch_dict[0], batch_dict[i]) 
            scores.append(cosine_sim * 100)

        row_result = {"Index": index, "Query10_Positive": scores[0]} 
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores[i]  

        all_results.append(row_result)


    results_df = pd.DataFrame(all_results)
    file_name = os.path.basename(data_path).replace(".csv", "_task4_results_new.csv")
    new_path = os.path.join(target_directory, file_name)
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
        gritlm_task1(data_path)

    for data_path in task2_data_paths:
        gritlm_task2(data_path)

    for data_path in task3_data_paths:
        gritlm_task3(data_path)

    for data_path in task4_data_paths:
        gritlm_task4(data_path)
