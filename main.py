import ollama
from huggingface_hub import whoami
from datasets import load_dataset
import numpy as np
from pandas import DataFrame
import os
import multiprocessing

from functions import get_prediction_and_confidence
import naive, cot, graphrag

from dotenv import dotenv_values
config = dotenv_values(".env")

dataset = load_dataset("lytang/LLM-AggreFact",token=config["HUGGINGFACE_ACCESS_TOKEN"])
models = []
for item in dataset["test"]:
    models.append(item["dataset"])
#print('models',list(dict.fromkeys(models)))

def naive_approach_prompt(item):
        
    response = naive.query_llm(item["doc"], item["claim"])
    
    id = item["contamination_identifier"]
    predicted, confidence = get_prediction_and_confidence(id, response)

    return [
        item["dataset"], #  dataset
        item["contamination_identifier"], # dataset:id
        item["label"], # class
        predicted, # predicted
        confidence, # confidence
    ]

def cot_approach_prompt(item):
    response = cot.query_llm(item["doc"], item["claim"])

    id = item["contamination_identifier"]
    predicted, confidence = get_prediction_and_confidence(id, response)

    return [
        item["dataset"], #  dataset
        item["contamination_identifier"], # dataset:id
        item["label"], # class
        predicted, # predicted
        confidence, # confidence
    ]

def graphrag_approach_prompt(item):
    id = item["contamination_identifier"]
    model_alias = "mixtral_8x7b"
    graph_path = f'output/{model_alias}/graphs/{id}.txt'
    graph = graphrag.get_graph_from_llm(ollama, model='mixtral:8x7b', options={"temperature": 0}, document=item["doc"], graph_path = graph_path)
    #print(f'Graph result for {item["contamination_identifier"]}', graph)
    response = graphrag.query_graph_llm(ollama, model='mixtral:8x7b', options={"temperature": 0}, graph=graph, claim=item["claim"])

    predicted, confidence = get_prediction_and_confidence(id, response)
    
    return [
        item["dataset"], #  dataset
        id, # dataset:id
        item["label"], # class
        predicted, # predicted
        confidence, # confidence
    ]

def document_and_graphrag_approach_prompt(item):
    id = item["contamination_identifier"]
    model_alias = "mixtral_8x7b"
    graph_path = f'output/{model_alias}/graphs/{id}.txt'
    graph = graphrag.get_graph_from_llm(ollama, model='mixtral:8x7b', options={"temperature": 0}, document=item["doc"], graph_path = graph_path)
    #print(f'Graph result for {item["contamination_identifier"]}', graph)
    response = graphrag.query_document_and_graph_llm(ollama, model='mixtral:8x7b', options={"temperature": 0}, document=item["doc"], graph=graph, claim=item["claim"])

    predicted, confidence = get_prediction_and_confidence(id, response)
    
    return [
        item["dataset"], #  dataset
        id, # dataset:id
        item["label"], # class
        predicted, # predicted
        confidence, # confidence
    ]

# setup experiment
experiments = [
    {
        "model": "mixtral:8x7b",
        "model_alias": "mixtral_8x7b",
        "approachs": [
            "Naive",
            "CoT",
            "GraphRAG",
            "RAG_and_GraphRAG"
        ],
        "datasets":[
            "AggreFact-CNN",
            "AggreFact-XSum",
            "TofuEval-MediaS",
            "TofuEval-MeetB",
            "Wice",
            #"Reveal",
            "ClaimVerify",
            #"FactCheck-GPT",
            "ExpertQA",
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
            elif approach == "GraphRAG":
                results.append(pool_obj.map(graphrag_approach_prompt,items))#items[0:3]
            elif approach == "RAG_and_GraphRAG":
                results.append(pool_obj.map(document_and_graphrag_approach_prompt,items))#items[0:3]
            pool_obj.close()
            
            for result in results[0]:
                result.insert(0, approach)
                result.insert(0, model)
           
            df = DataFrame(data=np.array(results[0]), 
                index=np.arange(len(results[0])), 
                columns=['model','approach','dataset','id','label','predicted','confidence'],
            )

            output_folder = f'output/{model_alias}/{approach}'
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            df.to_csv(f'{output_folder}/{dataset_name}.csv', index=False,header=False)  
            print(df)