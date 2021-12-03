from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("MyApp").getOrCreate()

df = spark.read.load("Collection_M")

