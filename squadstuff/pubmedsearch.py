import pandas as pd
from Bio import Entrez, Medline
import time

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from boolq.bert_modules import BoolQBertModule


def search(query):

    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax='100',
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    return results

def fetch_details(id_list):
    ids = ','.join(id_list)
    # Entrez.email = 'your.email@example.com'
    handle = Entrez.efetch(db='pubmed',
                           rettype="medline", retmode="text",
                           id=ids)
    results = Medline.parse(handle)
    return results
Entrez.email = 'amirvt92@gmail.com'
topics = pd.read_csv("./data/topics.csv", sep="\t")
#%%
all_papers = []
for q in tqdm(topics["query"]):
    results = search(q)
    id_list = results['IdList']
    papers = fetch_details(id_list)
    all_papers.append(list(papers))
    time.sleep(1)

#%%
import pickle
with open('./squadstuff/pubmedsearches.pkl', 'wb') as f:
    pickle.dump(all_papers, f)

#%%

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-base", num_labels=3).to(0)
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-TinyBERT-L6-v2")
# model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/paraphrase-TinyBERT-L6-v2",

lightning_module = BoolQBertModule(
    tokenizer=tokenizer,
    model=model,
    num_classes=3,
    train_metrics="Accuracy".split(),
    valid_metrics="Accuracy F1Score".split(),

)

lightning_module.load_from_checkpoint(
    "checkpoints/boolq-qrel/deberta-base-lr=1e-05-batch_size=16-aug/epoch=02-valid_F1=0.902-valid_Accuracy=0.902.ckpt",
    model=model,
    tokenizer=tokenizer,
)