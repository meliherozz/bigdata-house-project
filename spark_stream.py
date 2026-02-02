from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.sql.functions import from_json, col
from pyspark.ml import PipelineModel

spark = (
    SparkSession.builder
    .appName("HousePriceStreaming")
    .getOrCreate()
)

# Kafka'dan gelecek JSON için şema (4 feature)
schema = StructType([
    StructField("bathrooms", DoubleType(), True),
    StructField("floors", DoubleType(), True),
    StructField("lat", DoubleType(), True),
    StructField("long", DoubleType(), True),
])

kafka_df = (
    spark.readStream
         .format("kafka")
         .option("kafka.bootstrap.servers", "localhost:9092")
         .option("subscribe", "houses")
         .option("startingOffsets", "latest")
         .load()
)

json_df = kafka_df.selectExpr("CAST(value AS STRING) as json_str")

parsed_df = (
    json_df
    .select(from_json(col("json_str"), schema).alias("data"))
    .select("data.*")
)

feature_cols = ["bathrooms", "floors", "lat", "long"]

# Güvenlik: 4 feature'dan biri bile NULL ise satırı at
clean_df = parsed_df.na.drop(subset=feature_cols)

# Eğitilmiş pipeline model
model = PipelineModel.load("models/house_model")

predictions = model.transform(clean_df)

# Çıktıda hem input'u hem prediction'ı göster
output_df = predictions.select(
    "bathrooms",
    "floors",
    "lat",
    "long",
    "prediction"
)

query = (
    output_df.writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", "false")
        .option("checkpointLocation", "checkpoint/house_stream")
        .start()
)

query.awaitTermination()