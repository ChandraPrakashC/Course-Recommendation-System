# Course-Recommendation-System

This repository contains a **Course Recommendation System** built using **Apache Spark** and **Streamlit**. The system recommends courses to users based on their ratings and other interactions with the courses.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Technologies](#technologies)
- [Model](#model)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Course Recommendation System uses **Apache Spark**'s collaborative filtering capabilities to generate course recommendations for users. The recommendations are based on a matrix factorization approach using **ALS (Alternating Least Squares)** algorithm. The recommendations are visualized using **Streamlit** and **Plotly**.

The system consists of two main components:
1. **Streamlit Web Application** (`app.py`): This provides the user interface for users to input their details and view course recommendations.
2. **Streamlit Visualization Dashboard** (`streamlit.py`): This displays various metrics such as RMSE (Root Mean Squared Error), user recommendations, and course recommendations through interactive plots.

## Installation

### Prerequisites
To run this system, you will need:
- Python 3.7+
- Apache Spark (installed with PySpark)
- Streamlit
- Plotly
- Pandas

### Steps

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ChandraPrakashC/Course-Recommendation-System.git
    cd Course-Recommendation-System
    ```

2. **Install required packages**:
    Use `pip` to install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

    Alternatively, install the required packages individually:
    ```bash
    pip install pyspark streamlit plotly pandas
    ```

## Usage

### 1. Running the Course Recommendation System Web Application
To launch the **Streamlit Web Application** (`app.py`):
```bash
streamlit run app.py
The app will start on your local server, and you will be able to interact with it through your browser.

2. Running the Visualization Dashboard
To run the Streamlit Visualization Dashboard (streamlit.py):


streamlit run streamlit.py
This dashboard provides insights into the recommendations for users and courses, visualized with Plotly.

## Data
The data used for this recommendation system is in CSV format (udemy_courses.csv). It contains course data such as:

course_id: Unique identifier for each course
user_id: Unique identifier for each user
num_reviews: Number of reviews for each course
The file should be in the same directory as the script or you can modify the path in the code to match your file's location.

## Technologies
Apache Spark: For building and training the recommendation model using the ALS (Alternating Least Squares) algorithm.
Streamlit: For building the interactive web application and visualization dashboard.
Plotly: For generating interactive visualizations for course and user recommendations.
Pandas: For data manipulation and processing.
Model
The recommendation model uses the ALS (Alternating Least Squares) algorithm provided by PySpark's MLlib. The model is trained on the user-item ratings and predicts ratings for courses not yet rated by the user. It recommends the top 5 courses for each user.

## Visualization
Using Plotly and Streamlit, we visualize the following:

Top 5 recommended courses for each user: A bar plot of recommended courses for users.
Top 5 recommended users for each course: A bar plot of users who are most likely to enjoy a given course.

## Contributing
We welcome contributions to improve the system! Feel free to fork the repository and create a pull request. Here are some ways you can contribute:

Fix bugs or improve code efficiency.
Add new features or algorithms.
Enhance the visualizations.

## License
Our project is licensed under the MIT License - see the LICENSE file for details.

## Conclusion

The **Course Recommendation System** leverages the power of **Apache Spark** for efficient processing and model training, providing users with personalized course recommendations based on their interactions. By using **Alternating Least Squares (ALS)** for collaborative filtering, the system offers an intuitive and scalable way to generate suggestions for users, ensuring that the recommendations are relevant and accurate.

The **Streamlit** web application makes it easy to interact with the system, allowing users to input their ID and view course suggestions in real-time. Meanwhile, the **Streamlit Visualization Dashboard** provides insightful visualizations, including user and course recommendations, performance metrics like RMSE, and interactive plots to help analyze the recommendation patterns.

This system can be further enhanced with more advanced techniques in machine learning and deep learning, as well as extended to different domains like movie recommendations or product recommendations.

We hope this project serves as a useful tool for anyone interested in building a recommendation engine, whether for educational content or other personalized systems.
