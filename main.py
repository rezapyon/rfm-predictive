import glob
import os
import shutil

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, count, sum, max, lit
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

if __name__ == "__main__":
    spark = (SparkSession.builder.appName("E-commerce IDSS")
             .master("local[*]")
             .config("spark.driver.bindAddress", "127.0.0.1")
             .getOrCreate())

    data = spark.read.csv("./online_retail.csv", header=True, inferSchema=True)
    data = data.dropna()

    data = (data.withColumn("Quantity", col("Quantity").cast("int"))
            .withColumn("UnitPrice", col("UnitPrice").cast("double")))

    data = data.withColumn("FinalPrice", col("Quantity") * col("UnitPrice"))

    rfm_data = data.groupBy("CustomerID").agg(
        (datediff(lit("2012-01-01").cast("date"), max("InvoiceDate"))).alias("Recency"),
        count("InvoiceDate").alias("Frequency"),
        sum("FinalPrice").alias("Monetary"),
    )

    assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features")
    rfm_features = assembler.transform(rfm_data)

    # K-Means Clustering
    kmeans = KMeans().setK(4).setSeed(1)
    kmeans_model = kmeans.fit(rfm_features)
    rfm_data = kmeans_model.transform(rfm_features)

    final_results = rfm_data.select("CustomerID", "Recency", "Frequency", "Monetary", "prediction")

    output_dir = "./final_result"
    final_results.write.mode("overwrite").csv(output_dir, header=True)

    part_file = os.path.join(output_dir, "part-00000-*.csv")
    final_file = "./final_results.csv"

    for filename in glob.glob(part_file):
        shutil.move(filename, final_file)

    shutil.rmtree(output_dir)

    spark.stop()