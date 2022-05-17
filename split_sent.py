import re
import sys

import spacy
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *

spark = SparkSession.builder.appName("MyApp").getOrCreate()

if len(sys.argv) > 1:
    inp = sys.argv[1]
    out = sys.argv[2]
else:
    inp = "Top1kBM25_2019"
    out = "Top1kBM25_2019"

df = spark.read.load(f"./data/{inp}")
window_size, step = 1, 1


print(df.count())

schema = ArrayType(StringType())

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

#
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

lol_udf = udf(sentencize, schema)

df_new = df.withColumn("passage", lol_udf("text")).selectExpr("docno", "topic", "score as bm25",
                                                              "explode(passage) as passage", "url")

df_new.repartition(1).write.mode("overwrite").save(f"./data/{out}_1p_sentences")
