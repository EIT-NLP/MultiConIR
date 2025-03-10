import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_directory = 'MultiConIR/datasets/results/followir/'


model_name = "/code/.cache/huggingface/FollowIR-7B"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
token_false_id = tokenizer.get_vocab()["false"]
token_true_id = tokenizer.get_vocab()["true"]

template = """<s> [INST] You are an expert Google searcher, whose job is to determine if the following document is relevant to the query (true/false). Answer using only one word, one of those two choices.

Query: {query}
Document: {text}
Relevant (only output one word, either "true" or "false"): [/INST] """

def followir_task1(data_path, model, tokenizer, multiply_factor=100, normalize=False):
    df = pd.read_csv(data_path)
    
    file_name = os.path.basename(data_path).replace(".csv", "_task1_v2.csv")
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
            
            prompts = [
                template.format(query=query_text, text=positive_text),
                template.format(query=query_text, text=hn_text)
            ]
            
            tokens = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                pad_to_multiple_of=None
            ).to(device)

            with torch.no_grad():
                batch_scores = model(**tokens).logits[:, -1, :]
                true_vector = batch_scores[:, token_true_id]
                false_vector = batch_scores[:, token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                scores = batch_scores[:, 1].exp().tolist()

            if multiply_factor is not None:
                scores = [s * multiply_factor for s in scores]
            
            row_result[f"Query{i}_Positive"] = scores[0]
            row_result[f"Query{i}_HN1"] = scores[1]

        all_results.append(row_result)
        print(row_result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def followir_task2(data_path, model, tokenizer, multiply_factor=100, normalize=False):
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

        prompts = [
            template.format(query=query_text, text=positive_text)
        ] + [
            template.format(query=query_text, text=hn) for hn in hn_texts
        ]
        
        tokens = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            pad_to_multiple_of=None
        ).to(device)

        with torch.no_grad():
            batch_scores = model(**tokens).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()

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

def followir_task3(data_path, model, tokenizer, multiply_factor=100, normalize=False):
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

        prompts = [
            template.format(query=query_text, text=positive_text)
        ] + [
            template.format(query=query_text, text=hn) for hn in hn_texts
        ]

        tokens = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            pad_to_multiple_of=None
        ).to(device)

        with torch.no_grad():
            batch_scores = model(**tokens).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()

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

def followir_task4(data_path, model, tokenizer, multiply_factor=100, normalize=False):
    df = pd.read_csv(data_path)

    file_name = os.path.basename(data_path).replace(".csv", "_task4_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for index, row in df.iterrows():
        print(f"Processing row {index + 1}/{len(df)}")
        row_result = {"Index": index}

        query_text = row['Natural_Query10']
        positive_text = row['Positive']

        hn_texts = [row[f'HN{i}'] for i in range(1, 11)]

        prompts = [
            template.format(query=query_text, text=positive_text)
        ] + [
            template.format(query=query_text, text=hn) for hn in hn_texts
        ]

        tokens = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            pad_to_multiple_of=None
        ).to(device)

        with torch.no_grad():
            batch_scores = model(**tokens).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()

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
        followir_task1(
            data_path=data_path,
            model=model,
            tokenizer=tokenizer,
            multiply_factor=100,    
            normalize=False        
        )

    for data_path in task2_data_paths:
        followir_task2(
            data_path=data_path,
            model=model,
            tokenizer=tokenizer,
            multiply_factor=100,    
            normalize=False        
        )

    for data_path in task3_data_paths:
        followir_task3(
            data_path=data_path,
            model=model,
            tokenizer=tokenizer,
            multiply_factor=100,    
            normalize=False        
        )

    for data_path in task4_data_paths:
        followir_task4(
            data_path=data_path,
            model=model,
            tokenizer=tokenizer,
            multiply_factor=100,    
            normalize=False        
        )