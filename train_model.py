from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Spark Session oluştur
spark = SparkSession.builder.appName("HousePriceTrain").getOrCreate()

# CSV'yi oku
data = spark.read.csv("data/house_data.csv", header=True, inferSchema=True)

# Bu sefer SADECE bu 4 feature'ı kullanıyoruz
feature_cols = [
    "bathrooms",
    "floors",
    "lat",
    "long"
]

# price ve bu 4 feature'da NULL olan satırları at
data = data.na.drop(subset=["price"] + feature_cols)

# Feature vektörü
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

# Gradient Boosted Trees regressor
gbt = GBTRegressor(
    featuresCol="features",
    labelCol="price",
    maxDepth=5,
    maxIter=100
)

pipeline = Pipeline(stages=[assembler, gbt])

train_data, test_data = data.randomSplit([0.6, 0.4], seed=42)

model = pipeline.fit(train_data)

predictions = model.transform(test_data)

evaluator = RegressionEvaluator(
    labelCol="price",
    predictionCol="prediction",
    metricName="r2"
)
r2 = evaluator.evaluate(predictions)
print(f"R2 score: {r2}")

# Modeli kaydet
model.save("models/house_model")

spark.stop()