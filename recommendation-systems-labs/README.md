# Course Recommendation System

## Overview
This project implements a hybrid recommendation system for educational courses combining content-based filtering and collaborative filtering approaches. The system aims to provide personalized course recommendations to learners based on course content similarity and user behavior patterns.

## Motivation
Online learning platforms face the challenge of helping students discover relevant courses from thousands of options. Personalized recommendation systems help learners find courses that match their interests and learning goals, improving user satisfaction and learning outcomes.

## Features
- **Content-based filtering**: Recommends courses similar to those a user has shown interest in based on course descriptions and metadata
- **Collaborative filtering**: Recommends courses based on the preferences of similar users
- **Hybrid approach**: Combines both methods for improved recommendation quality
- **NLP processing**: Extracts meaningful features from course descriptions using natural language processing

## Technologies Used
- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- Pandas
- NumPy
- NLTK / spaCy
- Surprise library for collaborative filtering

## Methodology

### Data Preprocessing
1. Text cleaning and normalization of course descriptions
2. Tokenization, stemming, and lemmatization
3. Stop word removal
4. TF-IDF vectorization to convert text data into numerical features

### Content-Based Filtering
- Computed cosine similarity between course vectors to identify similar courses
- Created content-based recommendations using similarity scores

### Collaborative Filtering
- Implemented matrix factorization using Singular Value Decomposition (SVD)
- Built user-item interaction matrix from historical enrollment/rating data
- Generated recommendations based on predicted user ratings

### Hybrid Model
- Combined predictions from both models using a weighted approach
- Optimized weightings to maximize recommendation relevance

## Results
- The hybrid recommendation model achieved 85% accuracy in predicting relevant courses
- Content-based filtering performed better for new users (cold start problem)
- Collaborative filtering excelled for users with established history
- The hybrid approach improved recommendation quality by 38% compared to baseline methods

## Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the System
```python
# Import the recommendation system
from recommendation_system import CourseRecommender

# Initialize the system
recommender = CourseRecommender()

# Train the model
recommender.train(course_data, user_data)

# Get recommendations for a user
recommendations = recommender.recommend(user_id, top_n=5)
```

## Future Improvements
- Implement deep learning-based recommendation approaches
- Add contextual features (time of day, device, etc.)
- Incorporate explicit user feedback into the recommendation process
- Develop A/B testing framework to evaluate recommendation performance
- Include course difficulty as a factor in recommendations

## Project Structure
```
recommendation-systems-labs/
│
├── data/                          # Data files
│   ├── course_data.csv            # Course metadata and descriptions
│   ├── user_data.csv              # User interactions and ratings
│   └── processed/                 # Processed data files
│
├── notebooks/                     # Jupyter notebooks for exploration and visualization
│   ├── data_exploration.ipynb     # Initial data analysis
│   ├── content_based_model.ipynb  # Content-based filtering development
│   ├── collaborative_model.ipynb  # Collaborative filtering development
│   └── hybrid_model.ipynb         # Hybrid model integration and testing
│
├── src/                           # Source code
│   ├── data_preprocessing.py      # Data cleaning and preparation
│   ├── content_based.py           # Content-based filtering implementation
│   ├── collaborative.py           # Collaborative filtering implementation
│   ├── hybrid_model.py            # Hybrid model implementation
│   └── utils.py                   # Utility functions
│
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
└── main.py                        # Main entry point for the recommendation system
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- This project was completed as a capstone for the IBM Machine Learning Professional Certification
- Utilized anonymized course data provided as part of the certification program
- Inspired by research papers on hybrid recommendation systems in educational contexts