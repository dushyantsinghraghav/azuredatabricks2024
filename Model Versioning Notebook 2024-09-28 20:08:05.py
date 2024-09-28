# Databricks notebook source
# MAGIC %md
# MAGIC ## Reading the sensor hourly count dataset and finding the patterns.
# MAGIC Lets read the larget dataset file and then apply machine learning algorithms on it to do model versioning in git

# COMMAND ----------

df = spark.read.csv("dbfs:/mnt/input/Pedestrian_Counting_System_-_Monthly__counts_per_hour.csv", header=True, inferSchema=True)
display(df)

# COMMAND ----------

from pyspark.sql.functions import col, when

#df = df.withColumn("Date_Time", col("Date_Time").cast("string"))
df = df.withColumn("Above_Threshold", when(col("Hourly_Counts") > 2000, True).otherwise(False))
display(df)

# COMMAND ----------

from pyspark.sql.functions import col, when

df = df.withColumn("Above_Threshold", when(col("Hourly_Counts") > 2000, 1).otherwise(0))
display(df)

# COMMAND ----------

from pyspark.sql.functions import count, lit

# Count the rows where Above_Threshold is 1 and 0
count_above_threshold = df.filter(col("Above_Threshold") == 1).count()
count_below_threshold = df.filter(col("Above_Threshold") == 0).count()

# Calculate the ratio
ratio = count_above_threshold / count_below_threshold

# Create a DataFrame for plotting
imbalance_df = spark.createDataFrame([
    ("Above Threshold", count_above_threshold),
    ("Below Threshold", count_below_threshold)
], ["Category", "Count"])

# Display the ratio
print(f"Ratio of Above Threshold to Below Threshold: {ratio}")

# Plot the imbalance
display(imbalance_df)
display(df)

# COMMAND ----------

from pyspark.sql.types import IntegerType, LongType, FloatType, DoubleType

numeric_columns = [col for col in df.columns if col not in ["Above_Threshold", "Date_Time"] and isinstance(df.schema[col].dataType, (IntegerType, LongType, FloatType, DoubleType))]

summary_df = df.select(numeric_columns).summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
display(summary_df)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier, GBTClassifier
from pyspark.ml import Pipeline

# Select numeric columns
numeric_columns = [col for col in df.columns if col not in ["Above_Threshold", "Date_Time"] and isinstance(df.schema[col].dataType, (IntegerType, LongType, FloatType, DoubleType))]

# Create VectorAssembler for feature transformation
assembler = VectorAssembler(inputCols=numeric_columns, outputCol="features")

# Create Decision Tree and Gradient Boosted Tree classifiers
dt_classifier = DecisionTreeClassifier(labelCol="Above_Threshold", featuresCol="features")
gbt_classifier = GBTClassifier(labelCol="Above_Threshold", featuresCol="features")

# Create pipelines
dt_pipeline = Pipeline(stages=[assembler, dt_classifier])
gbt_pipeline = Pipeline(stages=[assembler, gbt_classifier])

# Explanation of hyperparameters for Decision Tree:
# - maxDepth: The maximum depth of the tree. Increasing maxDepth allows the model to learn more complex patterns but can lead to overfitting.
# - maxBins: The maximum number of bins used for discretizing continuous features. Increasing maxBins can improve the granularity of the model but also increases computational complexity.

# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql.functions import year, hour, to_timestamp, col

# Convert Date_Time to timestamp if it's not already
# Extract the hour from the Time column
#df2 = df.withColumn("Hour", hour("Time"))

# Filter data for training (2014-2018) and testing (2019) purposes
training_data = df.filter(
    (col("Year").between(2014, 2018)) & (col("Time").between(9, 23))
)
testing_data = df.filter(
    (col("Year") == 2019) & (col("Time").between(9, 23))
)

# Cache the training and testing data
training_data.cache()
testing_data.cache()

display(training_data)

# COMMAND ----------

print(training_data.count())
print(testing_data.count())

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# Define the feature columns
feature_columns = ["Year", "Time", "Sensor_ID", "Hourly_Counts"]

# Assemble the features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# Define the classifier
rf_classifier = RandomForestClassifier(labelCol="Above_Threshold", featuresCol="features")

# Create the pipeline
pipeline = Pipeline(stages=[assembler, rf_classifier])

# Train the model
model = pipeline.fit(training_data)

# Perform predictions on the testing data
predictions = model.transform(testing_data)

display(predictions)

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

# Define the decision tree classifier
dt_classifier = DecisionTreeClassifier(labelCol="Above_Threshold", featuresCol="features")

# Create the pipeline for decision tree
dt_pipeline = Pipeline(stages=[assembler, dt_classifier])

# Train the decision tree model
dt_model = dt_pipeline.fit(training_data)

# Perform predictions on the testing data using decision tree
dt_predictions = dt_model.transform(testing_data)

display(dt_predictions)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

# Initialize the evaluator
evaluator = RegressionEvaluator(labelCol="Above_Threshold", predictionCol="prediction", metricName="rmse")

# Calculate RMSE for Random Forest Classifier
rf_rmse = evaluator.evaluate(predictions)
print(f"Random Forest RMSE: {rf_rmse}")

# Calculate RMSE for Decision Tree Classifier
dt_rmse = evaluator.evaluate(dt_predictions)
print(f"Decision Tree RMSE: {dt_rmse}")

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col

# Function to display the count of each combination of label and prediction
def display_confusion_matrix(predictions, label_col, prediction_col):
    confusion_matrix = predictions.groupBy(label_col, prediction_col).count()
    display(confusion_matrix)

# Display confusion matrix for Decision Tree
display_confusion_matrix(dt_predictions, "Above_Threshold", "prediction")

# Assuming 'predictions' is the DataFrame for Random Forest predictions
display_confusion_matrix(predictions, "Above_Threshold", "prediction")

# Initialize evaluators
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="Above_Threshold", 
    rawPredictionCol="prediction", 
    metricName="areaUnderROC"
)
multiclass_evaluator = MulticlassClassificationEvaluator(
    labelCol="Above_Threshold", 
    predictionCol="prediction"
)

# Calculate metrics for Decision Tree
dt_auc = binary_evaluator.evaluate(dt_predictions)
dt_accuracy = multiclass_evaluator.evaluate(dt_predictions, {multiclass_evaluator.metricName: "accuracy"})
dt_recall = multiclass_evaluator.evaluate(dt_predictions, {multiclass_evaluator.metricName: "recallByLabel"})
dt_precision = multiclass_evaluator.evaluate(dt_predictions, {multiclass_evaluator.metricName: "precisionByLabel"})

# Calculate metrics for Random Forest
rf_auc = binary_evaluator.evaluate(predictions)
rf_accuracy = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "accuracy"})
rf_recall = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "recallByLabel"})
rf_precision = multiclass_evaluator.evaluate(predictions, {multiclass_evaluator.metricName: "precisionByLabel"})

# Display metrics
print(f"Decision Tree AUC: {dt_auc}, Accuracy: {dt_accuracy}, Recall: {dt_recall}, Precision: {dt_precision}")
print(f"Random Forest AUC: {rf_auc}, Accuracy: {rf_accuracy}, Recall: {rf_recall}, Precision: {rf_precision}")

# Determine the better model based on AUC
better_model = dt_model if dt_auc > rf_auc else rf_model

# Persist the better model
better_model.write().overwrite().save("/path/to/save/better_model")

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, FloatType
from pyspark.sql.functions import col

# Extract the Decision Tree model from the pipeline
dt_model = better_model.stages[-1]

# Print the leaf node splitting criteria
print("Leaf Node Splitting Criteria:")
print(dt_model.toDebugString)

# Get feature importances
feature_importances = dt_model.featureImportances

# Assuming the feature names are known and in the same order as the feature vector
feature_names = ["Year", "Month", "Mdate", "Day", "Time", "Sensor_ID", "Sensor_Name", "Hourly_Counts"]

# Define the schema
schema = StructType([
    StructField("Feature", StringType(), True),
    StructField("Importance", FloatType(), True)
])

# Create a DataFrame for feature importances
importances_df = spark.createDataFrame(
    [(feature_names[i], float(feature_importances[i])) for i in range(len(feature_importances))],
    schema
)

# Display the top-3 features with their importances
top_features = importances_df.orderBy(col("Importance").desc()).limit(3)
display(top_features)
