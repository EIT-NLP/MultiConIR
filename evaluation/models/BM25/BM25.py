import os
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm  
target_directory = 'MultiConIR/datasets/results/bm25/'

def preprocess(text):
    return text.lower().split()

def bm25_task1(data_path):

    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_bm25_task1.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        print(f'Processing row {index + 1}/{len(df)}')


        row_result = {"Index": index}

        for i in range(1, 11): 
            Query_col = f'Query{i}'
            Positive_col = f'Positive'
            HN_col = f'HN{i}'

            query_text = row[Query_col]
            positive_text = row[Positive_col]
            hn_text = row[HN_col]

            corpus = [positive_text, hn_text]
            tokenized_corpus = [preprocess(doc) for doc in corpus]

            bm25 = BM25Okapi(tokenized_corpus)

            tokenized_query = preprocess(query_text)
            scores = bm25.get_scores(tokenized_query)

            row_result[f"Query{i}_Positive"] = scores[0]  
            row_result[f"Query{i}_HN1"] = scores[1]     

            print(f"BM25 Scores for Query{i}: Positive={scores[0]}, HN={scores[1]}")

        all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def bm25_task2(data_path):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task2_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        print(f'Processing row {index + 1}/{len(df)}')

        row_result = {"Index": index}

        input_texts = [row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        tokenized_texts = [preprocess(text) for text in input_texts]

        bm25 = BM25Okapi(tokenized_texts)

        query_text = row['Query10']
        tokenized_query = preprocess(query_text)

        scores = bm25.get_scores(tokenized_query)

        row_result["Query10_Positive"] = scores[0]  
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores[i]  

        all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def bm25_task3(data_path):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task3_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        print(f'Processing row {index + 1}/{len(df)}')

        row_result = {"Index": index}

        input_texts = [row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        tokenized_texts = [preprocess(text) for text in input_texts]

        bm25 = BM25Okapi(tokenized_texts)

        query_text = row['Natural_Query10']
        tokenized_query = preprocess(query_text)

        scores = bm25.get_scores(tokenized_query)

        row_result["Query10_Positive"] = scores[0]  
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores[i]  

        all_results.append(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def bm25_task4(data_path):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task4_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
        print(f'Processing row {index + 1}/{len(df)}')

        row_result = {"Index": index}

        input_texts = [row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]

        tokenized_texts = [preprocess(text) for text in input_texts]

        bm25 = BM25Okapi(tokenized_texts)

        query_text = row['Query10']
        tokenized_query = preprocess(query_text)

        scores = bm25.get_scores(tokenized_query)

        row_result["Query10_Positive"] = scores[0]  
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores[i]  

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
        bm25_task1(data_path)

    for data_path in task2_data_paths:
        bm25_task2(data_path)

    for data_path in task3_data_paths:
        bm25_task3(data_path)   

    for data_path in task4_data_paths:
        bm25_task4(data_path) 