import glob
import os
import shutil

from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, current_date, count, when, to_date, sum, max
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType

if __name__ == "__main__":
    spark = (SparkSession.builder.appName("E-commerce IDSS")
             .master("local[*]")
             .config("spark.driver.bindAddress", "127.0.0.1")
             .getOrCreate())

    data = spark.read.csv("./ecommerce_dataset_updated.csv", header=True, inferSchema=True)
    data = data.dropna()

    data = data.withColumn("Final_Price", when(col("Final_Price").cast(DoubleType()).isNotNull(), col("Final_Price").cast(DoubleType())).otherwise(0.0))
    data = data.withColumn('Purchase_Date', to_date(col('Purchase_Date'), 'd-M-yyyy'))

    rfm_data = data.groupBy("User_ID").agg(
        (datediff(current_date(), max("Purchase_Date"))).alias("Recency"),
        count("Purchase_Date").alias("Frequency"),
        sum("Final_Price").alias("Monetary"),
    )

    # criteria: If Recency > 30 days and Frequency < 2, label as churned
    rfm_data = rfm_data.withColumn("Is_Churn", when((col("Recency") > 30) & (col("Frequency") < 2), 1).otherwise(0))

    assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features")
    rfm_features = assembler.transform(rfm_data)

    # K-Means Clustering
    kmeans = KMeans().setK(5).setSeed(1)
    kmeans_model = kmeans.fit(rfm_features)
    rfm_data = kmeans_model.transform(rfm_features)

    churn_data = rfm_data.select("User_ID", "Recency", "Frequency", "Monetary", "Is_Churn")

    # Vectorize features for logistic regression
    assembler = VectorAssembler(inputCols=["Recency", "Frequency", "Monetary"], outputCol="features")
    churn_features = assembler.transform(churn_data)

    # Train logistic regression model
    lr = LogisticRegression(featuresCol="features", labelCol="Is_Churn")
    lr_model = lr.fit(churn_features)

    # Make predictions
    predictions = lr_model.transform(churn_features)

    # Store results in a CSV file for Superset
    final_results = predictions.select("User_ID", "Recency", "Frequency", "Monetary", "Is_Churn", "prediction")

    output_dir = "./final_result"
    final_results.write.mode("overwrite").csv(output_dir, header=True)

    part_file = os.path.join(output_dir, "part-00000-*.csv")
    final_file = "./final_results.csv"

    for filename in glob.glob(part_file):
        shutil.move(filename, final_file)

    shutil.rmtree(output_dir)

    spark.stop()