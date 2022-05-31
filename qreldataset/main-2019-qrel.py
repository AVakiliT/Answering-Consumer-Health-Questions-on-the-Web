import gzip
import html
import os
import re
import sys

# import spacy
from lazynlp import parse_html, transliterate
from warcio import ArchiveIterator

corp_dir = "/project/6003284/smucker/group-data/corpora/ClueWeb12-B13/DiskB/"
# p = Path('/home/avakilit/group-data/corpora/ClueWeb12-B13/DiskB/').rglob('*.warc.gz')

from tqdm import tqdm

# for file in tqdm(p):
#     with open(file, 'rb') as f:
#         for record in tqdm(ArchiveIterator(f)):
#             url = record.rec_headers.get_header('WARC-Target-URI')
#
# #%%
# with open(file, 'rb') as f: stream = io.BytesIO(f.read())
# for record in tqdm(ArchiveIterator(stream)):
#     url = record.rec_headers.get_header('WARC-Target-URI')

# %%
import pandas as pd

qrels = pd.read_csv("./data/qrels/2019_qrels.txt", names="topic iter docno usefulness stance credibility".split(),
                    sep=" ")


def get_file(docno):
    (_, warc, warc_part, id) = docno.split("-")
    warc_file = f"{corp_dir}/ClueWeb12_{warc[:2]}/{warc}/{warc}-{warc_part}.warc.gz"
    return warc_file


qrels["warc_file"] = qrels.docno.apply(get_file)

n = int(sys.argv[1])
k = int(sys.argv[2])

qrels = qrels.iloc[n * k:n * k + k]


# nlp = spacy.blank("en")
# nlp.add_pipe("sentencizer")

def clean_page(page):
    page = page.decode('utf-8', errors='ignore')

    page = page.strip()
    if not page:
        return ''
    txt = parse_html(page)
    txt = transliterate(txt)
    txt = html.unescape(txt)
    return txt


# def sent_tokenize(page):
#     paragraphs = page.split("\n\n")
#     return ('\n\n'.join(
#         [' '.join(s.sent.sent.text.strip() for s in nlp(re.sub('\s+', " ", p.strip())).sents) for p in paragraphs]))


docs = []
for row in tqdm(qrels.itertuples(), total=qrels.shape[0]):

    with gzip.open(row.warc_file, 'rb') as stream:
        for i, record in enumerate(ArchiveIterator(stream)):
            if record.rec_type == 'response':
                trec_id = (record.rec_headers.get_header('WARC-TREC-ID'))
                uri = (record.rec_headers.get_header('WARC-Target-URI'))
                if trec_id == row.docno:
                    _html = record.content_stream().read()

                    # paragraphs = justext.justext(_html, justext.get_stoplist("English"))
                    text2 = clean_page(_html)
                    # text = sent_tokenize(clean_page(_html))

                    a = (row.topic, row.docno, row.usefulness, row.stance, row.credibility, text2, uri)
                    docs.append(a)
                    continue
    # if len(docs) == 2:
    #     break

df = pd.DataFrame(docs, columns="topic docno usefulness stance credibility text url".split())
out_path = f"/project/6003284/avakilit/Trec21_Data/2019qreldataset/2019qrels.parquet/{n:02d}.snappy.parquet"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_parquet(out_path)
