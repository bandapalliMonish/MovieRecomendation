# Databricks notebook source
# MAGIC %md ###Importing the SPARK SESSION

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Movie_Recomendation").getOrCreate()

# COMMAND ----------

# MAGIC %md ###Importing the Dataset

# COMMAND ----------

df = spark.read.csv(
    "/FileStore/tables/12_movielens_ratings.csv",
    inferSchema=True,
    header=True,
)

# COMMAND ----------

df.show()

# COMMAND ----------

# MAGIC %md ###Description of Dataset

# COMMAND ----------

df.describe().show()

# COMMAND ----------

# MAGIC %md ###Splitting the Dataset

# COMMAND ----------

training_set, test_set = df.randomSplit([0.8, 0.2])

# COMMAND ----------

# MAGIC %md ###Let's Create a Model

# COMMAND ----------

from pyspark.ml.recommendation import ALS
recommender = ALS(userCol= 'userId' , itemCol='movieId' , ratingCol='rating' )
recommender = recommender.fit(training_set)

# COMMAND ----------

# MAGIC %md ###Predicting Using The Test Set

# COMMAND ----------

prediction = recommender.transform(test_set)

# COMMAND ----------

prediction.show()

# COMMAND ----------

# MAGIC %md ###Evaluating the Model

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol='rating')
evaluator.evaluate(prediction)

# COMMAND ----------

# MAGIC %md ###Making a Recommendation

# COMMAND ----------

test_set.show() 

# COMMAND ----------

test_set.filter(test_set['userId']==23).show()

# COMMAND ----------

single_user = test_set.filter(test_set['userId']==23).select(['userId', 'movieId'])

# COMMAND ----------

single_user.show()


# COMMAND ----------

recommendations = recommender.transform(single_user)

# COMMAND ----------

recommendations.orderBy('prediction', ascending=False).show()

# COMMAND ----------


