import pandas as pd
import glob
import os

def get_result(row):
    y = row['label']
    predicted = '1' if row['response'] == 'True' else '0'
    #print (f'label: {y} - predicted: {predicted}')
    # 1 if result[0:5].strip() == 'True' else 0
    if y == predicted:
        return 1
    else:
        return 0
    
all_files = glob.glob("output/*/*/*.csv")
dfs = []
for filename in all_files:
    df = pd.read_csv(filename, names=['model','approach','dataset','id','label','response'], header=None)
    dfs.append(df)

df = pd.concat(dfs, axis=0, ignore_index=True)

df['result'] = df.apply(get_result, axis=1)

df = pd.pivot_table(df, values='result', index=['model','approach'],
                       columns=['dataset'], aggfunc={'result': ["sum","count"]}, fill_value=0).reset_index()
print(df.info()) 
print(df.columns) 
df.columns = [
    "Model"
    ,"Approach"
    #count
    ,"AggreFact-CNN_count"
    ,"AggreFact-XSum_count"
    ,"ClaimVerify_count"
    ,"ExpertQA_count"
    ,"FactCheck-GPT_count"
    ,"Lfqa_count"
    ,"Reveal_count"
    ,"TofuEval-MediaS_count"
    ,"TofuEval-MeetB_count"
    ,"Wice_count"
    ,"dataset_count"
    #sum
    ,"AggreFact-CNN_sum"
    ,"AggreFact-XSum_sum"
    ,"ClaimVerify_sum"
    ,"ExpertQA_sum"
    ,"FactCheck-GPT_sum"
    ,"Reveal_sum"
    ,"Lfqa_sum"
    ,"TofuEval-MediaS_sum"
    ,"TofuEval-MeetB_sum"
    ,"Wice_sum"
    ,"dataset_sum"
]
df['AggreFact-CNN_acc'] = df['AggreFact-CNN_sum'] / df['AggreFact-CNN_count']
df['AggreFact-XSum_acc'] = df['AggreFact-XSum_sum'] / df['AggreFact-XSum_count']
df['TofuEval-MediaS_acc'] = df['TofuEval-MediaS_sum'] / df['TofuEval-MediaS_count']
df['TofuEval-MeetB_acc'] = df['TofuEval-MeetB_sum'] / df['TofuEval-MeetB_count']
df['Wice_acc'] = df['Wice_sum'] / df['Wice_count']
df['Reveal_acc'] = df['Reveal_sum'] / df['Reveal_count']
df['ClaimVerify_acc'] = df['ClaimVerify_sum'] / df['ClaimVerify_count']
df['FactCheck-GPT_acc'] = df['FactCheck-GPT_sum'] / df['FactCheck-GPT_count']
df['ExpertQA_acc'] = df['ExpertQA_sum'] / df['ExpertQA_count']
df['Lfqa_acc'] = df['Lfqa_sum'] / df['Lfqa_count']

df = df[[
    "Model"
    ,"Approach"
    ,"AggreFact-CNN_count"
    ,"AggreFact-CNN_sum"
    ,"AggreFact-CNN_acc"
    ,"AggreFact-XSum_count"
    ,"AggreFact-XSum_sum"
    ,"AggreFact-XSum_acc"
    ,"TofuEval-MediaS_count"
    ,"TofuEval-MediaS_sum"
    ,"TofuEval-MediaS_acc"
    ,"TofuEval-MeetB_count"
    ,"TofuEval-MeetB_sum"
    ,"TofuEval-MeetB_acc"
    ,"Wice_count"
    ,"Wice_sum"
    ,"Wice_acc"
    ,"Reveal_count"
    ,"Reveal_sum"
    ,"Reveal_acc"
    ,"ClaimVerify_count"
    ,"ClaimVerify_sum"
    ,"ClaimVerify_acc"
    ,"FactCheck-GPT_count"
    ,"FactCheck-GPT_sum"
    ,"FactCheck-GPT_acc"
    ,"ExpertQA_count"
    ,"ExpertQA_sum"
    ,"ExpertQA_acc"
    ,"Lfqa_count"
    ,"Lfqa_sum"
    ,"Lfqa_acc"
]]

print(df)
df.to_csv(f'reports/mixtral_8x7b/result.csv')

###
#  llama3.2 - AggreFact-CNN (1.5% da amostra) - 559 rows
# 1º round - 75,84% accuracy
"""
        dataset   id  label  predicted
result                                
0           135  135    135        135
1           424  424    424        424
"""

# 2º round - 76,56% accuracy
"""
        dataset   id  label  predicted
result                                
0           131  131    131        131
1           428  428    428        428
"""

###
#  mixtral:8x7b - AggreFact-CNN (1.5% da amostra) - 559 rows
# 1º round - 88,01% accuracy
"""
result                                
0            67   67     67         67
1           492  492    492        492
"""

###
# FactCheck-GPT (5% da amostra) - 1567 rows
# 1º round - 63.81% accuracy
"""
result                                 
0           567   567    567        567
1          1000  1000   1000       1000
"""

###
# ClaimVerify (3.6% da amostra) - 1088 rows
# 1º round - 61.61% accuracy
"""
result                                
0           418  418    418        418
1           671  671    671        671
"""

###
#  mixtral:8x7b - ClaimVerify (3.6% da amostra) - 1089 rows
# 1º round - Referência 64.3 - resultado 72.63% accuracy
"""
result                                
0           298  298    298        298
1           791  791    791        791
"""

###
# llama3.2 - AggreFact-XSum (2.6% da amostra) - 559 rows
# 1º round - 68.10% accuracy
"""
result                                
0           179  179    179        179
1           380  380    380        380
"""

###
#  mixtral:8x7b - AggreFact-XSum (2.6% da amostra) - 559 rows
# 1º round - 69.76% accuracy
"""
result                                
0           169  169    169        169
1           390  390    390        390
"""

###
# Reveal (5.4% da amostra) - 1710 rows
# 1º round - 69.76% accuracy
"""
result                                 
0           518   518    518        518
1          1193  1193   1193       1193
"""

###
# ExpertQA (12.4% da amostra) -  3702rows
# 1º round - 70.2% accuracy
"""
result                                 
0          1101  1101   1101       1101
1          2602  2602   2602       2602
"""

