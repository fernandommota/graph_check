import ollama
from huggingface_hub import whoami
from datasets import load_dataset
import numpy as np
from pandas import DataFrame
import os
import multiprocessing

from dotenv import dotenv_values
config = dotenv_values(".env")

dataset = load_dataset("lytang/LLM-AggreFact",token=config["HUGGINGFACE_ACCESS_TOKEN"])
models = []
for item in dataset["test"]:
    models.append(item["dataset"])
#print('models',list(dict.fromkeys(models)))

def naive_query_llm(document, claim):
    #llama3.2, mixtral:8x7b
    response = ollama.chat(model='mixtral:8x7b', messages=[
    {
        'role': 'user',
        'content': f'''
            Context information is below.
            ---------------------
            {document}
            ---------------------
            Convert the given the context to a graph. Given the context as a graph and not prior knowledge, check if the follow claim is true or false. Response only with true or false.
            Claim: {claim}
        ''',
    },
    ])
    return response['message']['content']

def naive_approach_prompt(item):
        
    response = naive_query_llm(item["doc"], item["claim"])
    #print(f'Pid: {os.getpid()} - result for {item["contamination_identifier"]}', result[0:15])

    return [
        item["dataset"], #  dataset
        item["contamination_identifier"], # dataset:id
        item["label"], # class
        response.strip()[0:4] # predicted # 1 if result[0:5].strip() == 'True' else 0
    ]

def cot_query_llm(document, claim):
    #llama3.2, mixtral:8x7b
    response = ollama.chat(model='mixtral:8x7b', messages=[
    {
        'role': 'user',
        'content': f'''
            Convert the given the context to a graph. Given the context as a graph and not prior knowledge, check if the follow claim is true or false. Response only with true or false.
            Example:
                Context: John is male and has 32 years, he is marriage with Mary, female and has 28 years.
                Graph Data:
                    nodes: John, male, 32 years, Mary
                    relationships: 
                        - John -> marriage -> Mary
                        - John -> sex -> Male
                        - John -> age -> 32 years
                        - Mary -> marriage -> John
                        - Mary -> sex -> Female
                        - Mary -> age -> 28 years
                Claim: Mary has 30 years and is John's sister.
                Correct answer: False
            User prompt:
                Context information is below.
                ---------------------
                {document}
                ---------------------
                Claim: {claim}
        ''',
    },
    ])
    return response['message']['content']

def cot_approach_prompt(item):
        
    response = cot_query_llm(item["doc"], item["claim"])
    print(f'Pid: {os.getpid()} - result for {item["contamination_identifier"]}', response[0:150])

    return [
        item["dataset"], #  dataset
        item["contamination_identifier"], # dataset:id
        item["label"], # class
        response.strip()[0:4] # predicted # 1 if result[0:5].strip() == 'True' else 0
    ]

# setup experiment
experiments = [
    {
        "model": "mixtral:8x7b",
        "model_alias": "mixtral_8x7b",
        "approachs": [
            #"Naive",
            "CoT"
        ],
        "datasets":[
            "AggreFact-CNN",
            #"AggreFact-XSum",
            #"TofuEval-MediaS",
            #"TofuEval-MeetB",
            #"Wice",
            #"Reveal",
            #"ClaimVerify",
            #"FactCheck-GPT",
            #"ExpertQA",
            #"Lfqa",
            #"RAGTruth"
        ]
    }
]

for experiment in experiments:
    model = experiment["model"]
    model_alias = experiment["model_alias"]
    for approach in experiment["approachs"]:
        for dataset_name in experiment["datasets"]:
            items = list(filter(lambda item: item["dataset"] == dataset_name, dataset["test"]))
            pool_obj = multiprocessing.Pool(12)
            results = []
            if approach == "Naive":
                results.append(pool_obj.map(naive_approach_prompt,items))#items[0:3]
            elif approach == "CoT":
                results.append(pool_obj.map(cot_approach_prompt,items))#items[0:3]
            pool_obj.close()
            
            for result in results[0]:
                result.insert(0, approach)
                result.insert(0, model)
                
            df = DataFrame(data=np.array(results[0]), 
                        index=np.arange(len(results[0])), 
                        columns=['model','approach','dataset','id','label','response'])

            output_folder = f'output/{model_alias}/{approach}'
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            df.to_csv(f'{output_folder}/{dataset_name}.csv', index=False)  
            print(df)