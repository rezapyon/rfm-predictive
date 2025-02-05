import glob
import os
import shutil

from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, current_date, count, when, sum, max
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
    # data = data.withColumn('Purchase_Date', to_date(col('Purchase_Date'), 'd-M-yyyy'))

    rfm_data = data.groupBy("CustomerID").agg(
        (datediff(current_date(), max("InvoiceDate"))).alias("Recency"),
        count("InvoiceDate").alias("Frequency"),
        sum("FinalPrice").alias("Monetary"),
    )

    # criteria: If Recency > 30 days and Frequency < 2, label as churned
    rfm_data = rfm_data.withColumn("IsChurn", when((col("Recency") > 30) & (col("Frequency") < 2), 1).otherwise(0))

    assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features")
    rfm_features = assembler.transform(rfm_data)

    # K-Means Clustering
    kmeans = KMeans().setK(5).setSeed(1)
    kmeans_model = kmeans.fit(rfm_features)
    rfm_data = kmeans_model.transform(rfm_features)

    churn_data = rfm_data.select("CustomerID", "Recency", "Frequency", "Monetary", "IsChurn")

    # Vectorize features for logistic regression
    assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features")
    churn_features = assembler.transform(churn_data)

    # Train logistic regression model
    lr = LogisticRegression(featuresCol="features", labelCol="IsChurn")
    lr_model = lr.fit(churn_features)

    # Make predictions
    predictions = lr_model.transform(churn_features)

    # Store results in a CSV file for Superset
    final_results = predictions.select("CustomerID", "Recency", "Frequency", "Monetary", "IsChurn", "prediction")

    output_dir = "./final_result"
    final_results.write.mode("overwrite").csv(output_dir, header=True)

    part_file = os.path.join(output_dir, "part-00000-*.csv")
    final_file = "./final_results.csv"

    for filename in glob.glob(part_file):
        shutil.move(filename, final_file)

    shutil.rmtree(output_dir)

    spark.stop()