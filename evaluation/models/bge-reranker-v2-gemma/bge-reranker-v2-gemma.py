import os
import torch
import pandas as pd

from FlagEmbedding import FlagLLMReranker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_directory = 'MultiConIR/datasets/results/bge-reranker-v2-gemma/'

def bge_task1(data_path, reranker, multiply_factor=100, normalize=True):

    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task1_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []
    
    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{len(df)}")
        row_result = {"Index": index}

        for i in range(1, 11):
            query_text = row[f'Query{i}']
            positive_text = row['Positive']
            hn_text = row[f'HN{i}']
            
            scores = reranker.compute_score(
                [
                    [query_text, positive_text],
                    [query_text, hn_text]
                ],
                normalize=normalize
            )
            
            if multiply_factor is not None:
                scores = [s * multiply_factor for s in scores]
            
            row_result[f"Query{i}_Positive"] = scores[0]  # Query vs Positive
            row_result[f"Query{i}_HN1"] = scores[1]       # Query vs HN

        all_results.append(row_result)
        print(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def bge_task2(data_path, reranker, multiply_factor=100, normalize=True):

    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task2_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []
    
    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{len(df)}")
        row_result = {"Index": index}

        query_text = row['Query10']
        positive_text = row['Positive']
        
        hn_texts = [row[f'HN{i}'] for i in range(1, 11)]
        
        scores = reranker.compute_score(
            [[query_text, positive_text]] + [[query_text, hn] for hn in hn_texts],
            normalize=normalize
        )

        if multiply_factor is not None:
            scores = [s * multiply_factor for s in scores]
        
        row_result["Query10_Positive"] = scores[0]  # Query10 vs Positive
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores[i] 

        all_results.append(row_result)
        print(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def bge_task3(data_path, reranker, multiply_factor=100, normalize=True):

    df = pd.read_csv(data_path)
    
    file_name = os.path.basename(data_path).replace(".csv", "_task3_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []
    
    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{len(df)}")
        row_result = {"Index": index}

        query_text = row['Natural_Query10']
        positive_text = row['Positive']
        
        hn_texts = [row[f'HN{i}'] for i in range(1, 11)]
        
        scores = reranker.compute_score(
            [[query_text, positive_text]] + [[query_text, hn] for hn in hn_texts],
            normalize=normalize
        )

        if multiply_factor is not None:
            scores = [s * multiply_factor for s in scores]
        

        row_result["Query10_Positive"] = scores[0]  
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores[i] 

        all_results.append(row_result)
        print(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def bge_task4(data_path, reranker, multiply_factor=100, normalize=True):
    df = pd.read_csv(data_path)
    
    file_name = os.path.basename(data_path).replace(".csv", "_task4_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []
    
    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{len(df)}")
        row_result = {"Index": index}

        query_text = row['Query10']
        positive_text = row['Positive']
        
        hn_texts = [row[f'HN{i}'] for i in range(1, 11)]
        
        scores = reranker.compute_score(
            [[query_text, positive_text]] + [[query_text, hn] for hn in hn_texts],
            normalize=normalize
        )
        
        if multiply_factor is not None:
            scores = [s * multiply_factor for s in scores]
        

        row_result["Query10_Positive"] = scores[0]  
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores[i] 

        all_results.append(row_result)
        print(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")
if __name__ == '__main__':
    # 初始化 bge-reranker-v2-m3
    reranker = FlagLLMReranker('/code/.cache/huggingface/bge-reranker-v2-gemma', use_fp16=True, device=device)

    task1_data_paths = ['MultiConIR/datasets/Task1/Books_Task1.csv',
                    'MultiConIR/datasets/Task1/Legal Document_Task1.csv',
                    'MultiConIR/datasets/Task1/Medical Case_Task1.csv',
                    'MultiConIR/datasets/Task1/Movies_Task1.csv',
                    'MultiConIR/datasets/Task1/People_Task1.csv'

]

    # 依次处理各个 CSV
    for data_path in task1_data_paths:
        bge_task1(
            data_path=data_path,
            reranker=reranker,
            multiply_factor=100,    # 与原先的 "* 100" 保持一致
            normalize=True         # 如果想要 [0,1] 概率分数，则设 True
        )


    task2_data_paths = ['MultiConIR/datasets/Task2_&_3/Books_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Legal Document_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Medical Case_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Movies_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/People_Task2_&_3.csv'

    ]
    for data_path in task1_data_paths:
        bge_task2(
            data_path=data_path,
            reranker=reranker,
            multiply_factor=100,    # 与原先的 "* 100" 保持一致
            normalize=True         # 如果想要 [0,1] 概率分数，则设 True
        )


    task3_data_paths = ['MultiConIR/datasets/Task2_&_3/Books_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Legal Document_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Medical Case_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/Movies_Task2_&_3.csv',
                        'MultiConIR/datasets/Task2_&_3/People_Task2_&_3.csv'

    ]   
    for data_path in task1_data_paths:
        bge_task3(
            data_path=data_path,
            reranker=reranker,
            multiply_factor=100,    # 与原先的 "* 100" 保持一致
            normalize=True         # 如果想要 [0,1] 概率分数，则设 True
        )



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
        bge_task4(
            data_path=data_path,
            reranker=reranker,
            multiply_factor=100,    # 与原先的 "* 100" 保持一致
            normalize=True         # 如果想要 [0,1] 概率分数，则设 True
        )