import pandas as pd
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import random

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_directory = 'MultiConIR/datasets/results/bge-en-icl/'

# Helper Functions
def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Extracts the embedding of the last non-padded token in each sequence.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}'

def get_detailed_example(task_description: str, query: str, response: str) -> str:
    return f'<instruct>{task_description}\n<query>{query}\n<response>{response}'

def get_new_queries(queries, query_max_len, examples_prefix, tokenizer):
    inputs = tokenizer(
        queries,
        max_length=query_max_len - len(tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
            tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False
    )
    prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)['input_ids']
    suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
    new_max_length = (len(prefix_ids) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
    new_queries = tokenizer.batch_decode(inputs['input_ids'])
    for i in range(len(new_queries)):
        new_queries[i] = examples_prefix + new_queries[i] + '\n<response>'
    return new_max_length, new_queries

# Model and Tokenizer Initialization
def initialize_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('/code/.cache/huggingface/bge-en-icl')
    model = AutoModel.from_pretrained('/code/.cache/huggingface/bge-en-icl')
    model.eval()
    model.to(device)
    return tokenizer, model

def new_model_task1(data_path, tokenizer, model, examples_prefix, query_max_len):

    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task1_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    task_description = 'Given a document search query, retrieve relevant documents that match the query.'

    print("Randomly selecting and encoding shared other_positives...")
    random_indices = random.sample(range(len(df)), 7)  
    other_positives = [df.loc[idx, 'Positive'] for idx in random_indices]
    detailed_queries = [f"<instruct>{task_description}\n<query>{q}\n<response>" for q in other_positives]


    batch_dict = tokenizer(
        detailed_queries, max_length=query_max_len, padding=True, truncation=True, return_tensors='pt'
    )
    batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = model(**batch_dict)
        shared_other_embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        shared_other_embeddings = F.normalize(shared_other_embeddings, p=2, dim=1)

    print("Shared other_positives embeddings cached successfully.")


    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')
        row_result = {"Index": index}

        for i in range(1, 11):  
            Query_col = f'Query{i}'
            Positive_col = f'Positive'
            HN_col = f'HN{i}'
            HN2_col = f'HN_{i}'

            input_texts = [row[Query_col], row[Positive_col], row[HN_col], row[HN2_col]]
            detailed_queries = [f"<instruct>{task_description}\n<query>{q}\n<response>" for q in input_texts]

            batch_dict = tokenizer(
                detailed_queries, max_length=query_max_len, padding=True, truncation=True, return_tensors='pt'
            )
            batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)


            combined_embeddings = torch.cat([embeddings, shared_other_embeddings], dim=0)


            scores = torch.mm(combined_embeddings[0:1], combined_embeddings[1:].T) * 100
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



def new_model_task2(data_path, tokenizer, model, examples_prefix):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task2_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        # Prepare detailed instruct inputs
        task_description = 'Given a document search query, retrieve relevant documents that match the query.'
        queries = [row['Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]
        detailed_queries = [get_detailed_instruct(task_description, q) for q in queries]

        # Tokenize and prepare batch
        batch_dict = tokenizer(detailed_queries, max_length=8192, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity scores
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        scores_list = scores[0].tolist()
        print(f"Scores for row {index + 1}: {scores_list}")

        # Save scores
        row_result = {
            "Index": index,
            "Query10_Positive": scores_list[0]
        }
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores_list[i]

        all_results.append(row_result)

    # Save Results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")


def new_model_task3(data_path, tokenizer, model):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task3_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        # Prepare detailed instruct inputs
        task_description = 'Given a document search query, retrieve relevant documents that match the query.'
        queries = [row['Natural_Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]
        detailed_queries = [get_detailed_instruct(task_description, q) for q in queries]

        # Tokenize and prepare batch
        batch_dict = tokenizer(detailed_queries, max_length=1024, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity scores
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        scores_list = scores[0].tolist()
        print(f"Scores for row {index + 1}: {scores_list}")

        # Save scores
        row_result = {
            "Index": index,
            "Query10_Positive": scores_list[0]
        }
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores_list[i]

        all_results.append(row_result)

    # Save Results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")

def new_model_task4(data_path, tokenizer, model):
    df = pd.read_csv(data_path)
    file_name = os.path.basename(data_path).replace(".csv", "_task4_results.csv")
    new_path = os.path.join(target_directory, file_name)
    os.makedirs(os.path.dirname(new_path), exist_ok=True)

    all_results = []

    for index, row in df.iterrows():
        print(f'Processing row {index + 1}/{len(df)}')

        # Prepare detailed instruct inputs
        task_description = 'Given a document search query, retrieve relevant documents that match the query.'
        queries = [row['Query10'], row['Positive']] + [row[f'HN{i}'] for i in range(1, 11)]
        detailed_queries = [get_detailed_instruct(task_description, q) for q in queries]

        # Tokenize and prepare batch
        batch_dict = tokenizer(detailed_queries, max_length=1024, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity scores
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        scores_list = scores[0].tolist()
        print(f"Scores for row {index + 1}: {scores_list}")

        # Save scores
        row_result = {
            "Index": index,
            "Query10_Positive": scores_list[0]
        }
        for i in range(1, 11):
            row_result[f"Query10_HN{i}"] = scores_list[i]

        all_results.append(row_result)

    # Save Results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(new_path, index=False)
    print(f"Results saved to {new_path}")


tokenizer, model = initialize_model_and_tokenizer()


examples = [
        {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
         'query': 'what is a virtual interface',
         'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."},
        {'instruct': 'Given a web search query, retrieve relevant passages that answer the query.',
         'query': 'causes of back pain in female for a week',
         'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."}
    ]
examples = [get_detailed_example(e['instruct'], e['query'], e['response']) for e in examples]
examples_prefix = '\n\n'.join(examples) + '\n\n'  # If there are examples


task1_data_paths = ['MultiConIR/datasets/Task1/Books_Task1.csv',
                    'MultiConIR/datasets/Task1/Legal Document_Task1.csv',
                    'MultiConIR/datasets/Task1/Medical Case_Task1.csv',
                    'MultiConIR/datasets/Task1/Movies_Task1.csv',
                    'MultiConIR/datasets/Task1/People_Task1.csv'

]

for data_path in task1_data_paths:
    new_model_task1(data_path, tokenizer, model)

task2_data_paths = ['MultiConIR/datasets/Task2_&_3/Books_Task2_&_3.csv',
                    'MultiConIR/datasets/Task2_&_3/Legal Document_Task2_&_3.csv',
                    'MultiConIR/datasets/Task2_&_3/Medical Case_Task2_&_3.csv',
                    'MultiConIR/datasets/Task2_&_3/Movies_Task2_&_3.csv',
                    'MultiConIR/datasets/Task2_&_3/People_Task2_&_3.csv'

]
for data_path in task2_data_paths:
    new_model_task3(data_path, tokenizer, model)

task3_data_paths = ['MultiConIR/datasets/Task2_&_3/Books_Task2_&_3.csv',
                    'MultiConIR/datasets/Task2_&_3/Legal Document_Task2_&_3.csv',
                    'MultiConIR/datasets/Task2_&_3/Medical Case_Task2_&_3.csv',
                    'MultiConIR/datasets/Task2_&_3/Movies_Task2_&_3.csv',
                    'MultiConIR/datasets/Task2_&_3/People_Task2_&_3.csv'

]

for data_path in task3_data_paths:
    new_model_task3(data_path, tokenizer, model)


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

for data_path in task4_data_paths:
    new_model_task4(data_path, tokenizer, model)