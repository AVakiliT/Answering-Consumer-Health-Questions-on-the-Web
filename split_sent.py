from nltk import sent_tokenize
from pyspark.sql.functions import udf

window_size, step = 6,  3
from pyspark.sql.types import *

df = spark.read.load("Top1kBM25_32p")

from pyspark.sql.types import *
schema = ArrayType(StringType())

window_size = 6
step = 3
from nltk import sent_tokenize

def lol(s):
      # TODO handle too small documents
     seq = sent_tokenize(s)
     return [' '.join(seq[i: i + window_size]) for i in range(0, len(seq) - window_size + 1, step)]

lol_udf = udf(lol,schema)

df.limit(10).withColumn("lol", lol_udf("text")).show()

df_new = df.withColumn("passage", lol_udf("text")).selectExpr("docno", "topic", "score as bm25", "explode(passage) as passage")

df_new.repartition(1).write.mode("overwrite").save("Top1kBM25_1p_passages")