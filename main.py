import ollama
from huggingface_hub import whoami
from datasets import load_dataset
import numpy as np
from pandas import DataFrame
import os
import multiprocessing

user = whoami(token="")

dataset = load_dataset("lytang/LLM-AggreFact")

def query_llm(document, claim):
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
#query_llm()

#print(dataset)
#print(dir(dataset["test"]))
#print(len(dataset["test"]))

def rag_item(item):
        
    result = query_llm(item["doc"], item["claim"])
    print(f'Pid: {os.getpid()} - result for {item["contamination_identifier"]}', result[0:15])

    return [
        item["dataset"], #  dataset
        item["contamination_identifier"], # dataset:id
        item["label"], # class
        result # predicted # 1 if result[0:5].strip() == 'True' else 0
    ]

items = list(filter(lambda item: item["dataset"] == "ClaimVerify", dataset["test"]))
pool_obj = multiprocessing.Pool(12)
results = []
results.append(pool_obj.map(rag_item,items))#items[0:3]
pool_obj.close()

print(f'results len: {len(results[0])}')
print(f'results', results[0])
df = DataFrame(data=np.array(results[0]), 
               index=np.arange(len(results[0])), 
               columns=['dataset','id','label','predicted'])

df.to_csv('output.csv', index=False)  
print(df)