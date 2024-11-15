import pandas as pd

def get_result(row):
    y = row['label']
    predicted = '1' if row['predicted'].strip()[0:4] == 'True' else '0'
    print (f'label: {y} - predicted: {predicted}')
    # 1 if result[0:5].strip() == 'True' else 0
    if y == predicted:
        return 1
    else:
        return 0
    
df = pd.read_csv('output.csv', names=['dataset','id','label','predicted'], header=None)

df['result'] = df.apply(get_result, axis=1)

print(df)  


print(df.groupby(['result']).count()) 

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

