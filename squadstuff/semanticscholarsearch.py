import time

import pandas as pd
import requests
from tqdm import tqdm
import json

# %%
topics = pd.read_csv("./data/topics_fixed.tsv.txt", sep="\t")
# %%
all_papers = []
for row in tqdm(topics.iterrows(), total=topics.shape[0]):
    q = row[1]["query"]
    t = row[1].topic
    url = f"http://api.semanticscholar.org/graph/v1/paper/search?query={'+'.join(q.split())}&limit=100&fields=title,authors,abstract,year,citationCount,influentialCitationCount"
    response = requests.get(url)
    time.sleep(2)
    all_papers.append((t, json.loads(response.content)))

# %%
datas = []
for topic, topic_papers in tqdm(all_papers):
    for data in topic_papers['data']:
        datas.append((topic, *data.values()))

df = pd.DataFrame(datas,
                  columns=['topic', 'paperId', 'title', 'abstract', 'year', 'citationCount', 'influentialCitationCount',
                           'authors'])

# %%
stuff = []
# def f(idx):
for idx in tqdm(df.paperId):
    url = f"https://api.semanticscholar.org/graph/v1/paper/{idx}?fields=fieldsOfStudy,tldr,embedding"
    response = requests.get(url)
    time.sleep(.1)
    x = json.loads(response.content)
    stuff.append({"paperId": idx, **x})
