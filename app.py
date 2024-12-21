import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col
from pyspark.sql import functions as F

# Initialize Spark session
spark = SparkSession.builder.appName("CourseRecommendationSystem").getOrCreate()

# Load the data
data = spark.read.csv('/content/udemy_courses.csv', header=True, inferSchema=True)

# Data Preprocessing: Convert data types if needed
data = data.withColumn("user_id", F.monotonically_increasing_id())
data = data.withColumn("course_id", col("course_id").cast("integer"))

# Get the max number of reviews to normalize the rating
max_reviews = data.agg(F.max("num_reviews")).collect()[0][0]

# Generate mock ratings based on the number of reviews (normalized to a 1-5 scale)
data = data.withColumn("rating", (col("num_reviews") / max_reviews * 5).cast("float"))

# Split the data into training and test sets
(training, test) = data.randomSplit([0.8, 0.2])

# Create ALS model
als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol="user_id",
    itemCol="course_id",
    ratingCol="rating",
    coldStartStrategy="drop"
)

# Train the model
model = als.fit(training)

# Streamlit UI for selecting a user
st.title('Course Recommendation System')
user_id = st.number_input('Enter User ID:', min_value=1, step=1)

if user_id:
    # Generate recommendations for the user
    user_recommendations = model.recommendForAllUsers(numItems=5)
    recommendations = user_recommendations.filter(col("user_id") == user_id).collect()

    if recommendations:
        recs_list = recommendations[0]["recommendations"]
        course_ids = [rec[0] for rec in recs_list]
        ratings = [rec[1] for rec in recs_list]

        # Display recommendations
        st.write(f"Top course recommendations for User {user_id}:")
        for course_id, rating in zip(course_ids, ratings):
            st.write(f"Course ID: {course_id}, Predicted Rating: {rating}")
    else:
        st.write(f"No recommendations found for User {user_id}.")

# Stop the Spark session
spark.stop()
