import re

import pandas as pd
import spacy
from tqdm import tqdm

tqdm.pandas()

window_size, step = 6, 3

df = pd.read_parquet("./qreldataset/2019qrels.parquet/")
print(df.count())


nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")


# def tokenize_windows(s):
#     s = re.sub('\s+', " ", s.strip())
#     doc = nlp(s)
#     sentences = [sent.sent.text.strip() for sent in doc.sents if len(sent) > 5]
#     tokens = nlp(' '.join(sentences))
#
#     if len(tokens) <= window_size:
#         return tokens.text.strip()
#     return [tokens[i: i + window_size].text.strip() for i in range(0, len(tokens), step)]


def sentencize(s):
    s = re.sub('\s+', " ", s.strip())
    sentences = [sent.sent.text.strip() for sent in nlp(s).sents if len(sent) > 3]
    if len(sentences) <= window_size:
        return [s]
    return [' '.join(sentences[i: i + window_size]) for i in range(0, len(sentences), step)]

df["passage"] = df.text.progress_apply(sentencize)
df_new = df.explode("passage")
df_new.to_parquet("./qreldataset/2019qrels.passages.parquet")
