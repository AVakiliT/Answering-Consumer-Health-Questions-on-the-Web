#%%
import pandas as pd
import xml.etree.cElementTree as et

from torchvision.datasets import MNIST

df = pd.read_parquet("./data/Top1kBM25_2019_1p_passages")

#%%
topics_head = ['topic', 'query', 'cochranedoi', 'description', 'narrative']
xml_root = et.parse("./data/2019topics.xml")
rows = xml_root.findall('topic')
xml_data = [
    [int(row.find('number').text), row.find('query').text, row.find('cochranedoi').text,
     row.find('description').text, row.find('narrative').text] for row in rows]
topics = pd.DataFrame(xml_data, columns=topics_head)
topics_answers = pd.read_csv("./data/2019topics_efficacy.txt", header=None, sep=' ',
                             names=['topic', 'efficacy'])
topics = pd.merge(topics, topics_answers, how='left', on='topic')

#%%
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import torch
torch.set_grad_enabled(False)
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
#%%
from datasets import load_dataset, Dataset

ds = load_dataset('crime_and_punish', split='train[:100]')
ds_with_embeddings = ds.map(lambda example: {'embeddings': ctx_encoder(**ctx_tokenizer(example["line"], return_tensors="pt"))[0][0].numpy()})
ds_with_embeddings.add_faiss_index(column='embeddings')

#%%
from datasets import load_dataset
ds = load_dataset("parquet",
                  data_files={
                      "train": "data/Top1kBM25_2019_1p_passages/part-00000-9e1561a2-6119-46ab-8c19-9076394ba1dd-c000.snappy.parquet"
                  },
                  split='train')
ds = ds.shard(num_shards=8, index=0, contiguous=True)
ds_with_embeddings = ds.map(
    lambda example: {
        'embeddings': ctx_encoder(**ctx_tokenizer(example["passage"], return_tensors="pt", truncation=True, padding=True).to(device=ctx_encoder.device))[0].cpu().numpy()
    }
                        , batched=True, batch_size=128)

#%%