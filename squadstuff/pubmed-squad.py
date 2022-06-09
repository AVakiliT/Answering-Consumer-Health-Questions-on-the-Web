import pickle

import pandas as pd
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained('test-squad-trained')

with open('../squadstuff/pubmedsearches.pkl', 'rb') as f:
    all_papers = pickle.load(f)
#%%
topics = pd.read_csv("./data/topics.csv", sep="\t")
abstracts = []
for topic, topic_papers in zip(topics["topic"], all_papers):
    for paper in topic_papers:
        if 'AB' in paper.keys():
            abstracts.append((topic, paper['AB']))

abstracts = pd.DataFrame(abstracts, columns="topic ab stract".split())