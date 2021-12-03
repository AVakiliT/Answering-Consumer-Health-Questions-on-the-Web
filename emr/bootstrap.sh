#!/bin/bash -xe

aws emr create-cluster --name test-emr-cluster --use-default-roles --release-label emr-5.28.0 --instance-count 3 --instance-type m3.xlarge --applications Name=JupyterHub Name=Spark Name=Hadoop --ec2-attributes KeyName=emr_cluster  --log-uri s3://s3-for-emr-cluster/

# Remember to enable ssh in inbound rules of cluster master
aws emr ssh --cluster-id j-ZE2GHW8SL7IN --key-pair-file .\emr_cluster.pem

scp -i .\emr_cluster.pem "C:\Users\Amir\playground\cc-index-table\target\cc-index-table-0.2-SNAPSHOT-jar-with-dependencies.jar" ec2-3-87-129-78.compute-1.amazonaws.com:~


spark-submit \
   --conf spark.hadoop.parquet.enable.dictionary=true \
   --conf spark.hadoop.parquet.enable.summary-metadata=false \
   --conf spark.sql.hive.metastorePartitionPruning=true \
   --conf spark.sql.parquet.filterPushdown=true \
   --conf spark.sql.parquet.mergeSchema=true \
   --class org.commoncrawl.spark.examples.CCIndexWarcExport $APPJAR \
   --csv s3://amirvt/Collection_M_CSV/2021/11/23/e13de7e2-b689-4e99-afa1-49c0b011baf5.csv \
   --numOutputPartitions 12 \
   --numRecordsPerWarcFile 20000 \
   s3://commoncrawl/cc-index/table/cc-main/warc/ \
   ./my_output_path/