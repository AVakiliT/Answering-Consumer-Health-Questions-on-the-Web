from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType, StructType, StructField, StringType, FloatType, ArrayType
spark = SparkSession.builder.appName("MyApp").getOrCreate()

c4 = spark.read.load("/home/avakilit/group-data/c4-parquet/")
schema=StructType([
    StructField("topic", IntegerType(), True),
    StructField("_", IntegerType(), True),
    StructField("docno", StringType(), True),
    StructField("usefulness", IntegerType(), True),
    StructField("stance", IntegerType(), True),
    StructField("credibility", IntegerType(), True)
])


qrels = spark.read.csv("/project/6004803/avakilit/Trec21_Data/data/qrels/2021_qrels.txt", sep=" ", schema=schema)


@udf(StringType())
def f(s):
    return f"c4-{int(s[21:25]):04}-{int(s.split('.')[-1]):06}"
# df.withColumn("docno2", f(col("docno")))
qrels = qrels.withColumn("docno", f(col("docno")))
# df.withColumnRenamed("docno2", f(col("docno")))
df=c4.join(qrels.select(*"topic docno usefulness stance credibility".split()), "docno", "inner")

df.write.save("/project/6004803/avakilit/Trec21_Data/data/qrel_2021", mode="overwrite")

#%%
import re
import sys

import spacy
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import *

# df = spark.read.load("/project/6004803/avakilit/Trec21_Data/data/qrel_2021")
df = spark.read.parquet(
    "/project/6004803/avakilit/Trec21_Data/data/Top1kBM25_2019",
    "/project/6004803/avakilit/Trec21_Data/data/Top1kBM25")


window_size, step = 6, 3


# print(df.count())
nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")

schema = ArrayType(StructType([
    StructField("passage_index",IntegerType()),
    StructField("passage",StringType())
]))

def sentencize(s):
    s = re.sub('\s+', " ", s.strip())
    sentences = [sent.sent.text.strip() for sent in nlp(s).sents if len(sent) > 3]
    if len(sentences) <= window_size:
        return [(0,s)]
    return [(i,' '.join(sentences[i: i + window_size])) for i in range(0, len(sentences), step)]

lol_udf = udf(sentencize, schema)
df_new = df.withColumn("passage", lol_udf("text"))

# df_new = df_new.selectExpr("topic,docno,timestamp,url,usefulness,stance,credibility,explode(passage) as passage".split(','))
# df_new = df_new.selectExpr('topic,docno,timestamp,url,usefulness,stance,credibility,passage["passage_index"] as passage_index,passage["passage"] as passage'.split(','))
# df_new.repartition(1).write.save(f"/project/6004803/avakilit/Trec21_Data/data/qrels.2021.passages_{window_size}_{step}", mode="overwrite")

df_new = df_new.selectExpr('topic,docno,timestamp,url,bm25 as bm25,explode(passage) as passage'.split(','))
df_new = df_new.selectExpr('topic,docno,timestamp,url,bm25 as bm25,passage["passage_index"] as passage_index,passage["passage"] as passage'.split(','))
df_new.repartition(1).write.save(f"/project/6004803/avakilit/Trec21_Data/data/RunBM25.1k.passages_{window_size}_{step}", mode="overwrite")

#%%
