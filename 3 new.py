import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

transactions = [
    ['bread','milk','beer'],
    ['bread','diapers','eggs','beer','colt'],
    ['milk','diapers','beer','cola'],
    ['bread','milk','diapers','beer'],
    ['bread','milk','diapers','cola'],
    ['bread','milk','beer'],
    ['bread','diapers','egg','beer','colt'],
    ['milk','diapers','beer','cola'],
    ['bread','milk','diapers','beer'],
    ['bread','milk','diapers','cola']
    ]

df=pd.DataFrame(transactions)

print("Original Dataset :- ")
print(df)

oneshot=pd.get_dummies(df)

frequent_itemsets = apriori(oneshot,min_support=0.5,use_colnames=True)
rules=association_rules(frequent_itemsets, min_threshold=0.75)

print("Apriori for min support=0.5 and confidence=0.75")
df2=pd.DataFrame(frequent_itemsets)
print("Frequent Itemsets :- ")
print(df2)
print("Association rules :-")
print(rules)


print("Apriori for min support=0.6 and confidence=0.6")
frequent_itemsets = apriori(oneshot,min_support=0.6,use_colnames=True)
rules=association_rules(frequent_itemsets, min_threshold=0.6)
print("Frequent Itemsets :- ")
print(frequent_itemsets)
print("Association rules :-")
print(rules)
