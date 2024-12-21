import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
import pandas as pd
import plotly.express as px

# Initialize Spark session
spark = SparkSession.builder.appName("CourseRecommendationSystem").getOrCreate()

# Streamlit Title
st.title("Course Recommendation System")

# Load the data
@st.cache_data
def load_data():
    # Replace 'path_to_data.csv' with your actual file path
    return spark.read.csv('path_to_data.csv', header=True, inferSchema=True)

data = load_data()

# Show raw data
if st.checkbox("Show raw data"):
    st.write(data.show(5))

# Data Preprocessing
data = data.withColumn("user_id", col("user_id").cast("integer"))
data = data.withColumn("course_id", col("course_id").cast("integer"))
data = data.withColumn("rating", col("rating").cast("float"))

# Split data into training and test sets
training, test = data.randomSplit([0.8, 0.2])

# Train ALS model
als = ALS(maxIter=10, regParam=0.1, userCol="user_id", itemCol="course_id", ratingCol="rating", coldStartStrategy="drop")
model = als.fit(training)

# Make predictions
predictions = model.transform(test)

# Evaluate the model using RMSE
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)

# Display RMSE
st.write(f"Root Mean Squared Error (RMSE): {rmse}")

# Generate recommendations
user_recommendations = model.recommendForAllUsers(numItems=5)
course_recommendations = model.recommendForAllItems(numUsers=5)

# Convert recommendations to Pandas DataFrame for visualization
def get_recommendations_df(recommendations, type_col, rec_col):
    recommendations_df = recommendations.select(type_col, rec_col).toPandas()
    recommendations_expanded = recommendations_df.explode(rec_col)
    recommendations_expanded[[f"{rec_col}_id", f"{rec_col}_rating"]] = pd.DataFrame(recommendations_expanded[rec_col].tolist(), index=recommendations_expanded.index)
    recommendations_expanded.drop(columns=[rec_col], inplace=True)
    return recommendations_expanded

user_recommendations_df = get_recommendations_df(user_recommendations, "user_id", "recommendations")
course_recommendations_df = get_recommendations_df(course_recommendations, "course_id", "recommendations")

# Plot the recommendations using Plotly
st.subheader("Top 5 Recommended Courses for Each User")
if st.checkbox("Show user recommendations"):
    st.write(user_recommendations_df.head(10))  # Display the first few user recommendations

    # Plot user recommendations
    user_plot = px.bar(user_recommendations_df.head(10), x="user_id", y="recommendations_rating",
                       color="recommendations_id", barmode="group",
                       title="Top Course Recommendations for Users")
    st.plotly_chart(user_plot)

st.subheader("Top 5 Recommended Users for Each Course")
if st.checkbox("Show course recommendations"):
    st.write(course_recommendations_df.head(10))  # Display the first few course recommendations

    # Plot course recommendations
    course_plot = px.bar(course_recommendations_df.head(10), x="course_id", y="recommendations_rating",
                         color="recommendations_id", barmode="group",
                         title="Top User Recommendations for Courses")
    st.plotly_chart(course_plot)

# Stop the Spark session when the app closes
spark.stop()
