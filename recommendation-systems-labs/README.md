# Machine Learning Recommendation Systems

This repository contains a comprehensive series of Jupyter notebooks that demonstrate the implementation of various recommendation system techniques, from basic content-based approaches to advanced neural network methods.

## Project Overview

The project explores recommendation systems within the context of an online learning platform, using course enrollment data to predict and suggest relevant courses to users. The implementation progresses from exploratory data analysis through increasingly sophisticated recommendation algorithms.

## Dataset

The analysis uses two primary datasets:
- **Course metadata**: 307 unique courses with binary indicators for 14 different genres
- **User enrollment data**: 233,306 ratings from 33,901 users

## Implementation Approaches

The project demonstrates multiple recommendation system paradigms:

### Content-Based Filtering
- **Text Analysis**: Converts course titles and descriptions into numeric feature vectors using Bag of Words techniques
- **Similarity Metrics**: Implements cosine similarity, Euclidean distance, and Jaccard similarity for course comparison
- **User Profiling**: Generates user interest profiles by combining course ratings with genre classifications
- **Vector Operations**: Uses dot product operations between user profiles and course features to generate recommendations

### Collaborative Filtering
- **K-Nearest Neighbors**: Implements memory-based collaborative filtering using both user-based and item-based approaches
- **Matrix Factorization**: Applies Non-negative Matrix Factorization to decompose the user-item matrix into latent factors
- **Clustering**: Groups users with similar interests using K-means and leverages group patterns for recommendations

### Deep Learning
- **Neural Embeddings**: Implements a neural network architecture that learns compact representations of users and items
- **Dot Product Network**: Creates a specialized network architecture that mimics matrix factorization through neural components
- **Latent Feature Extraction**: Extracts learned embedding vectors for downstream recommendation tasks

## Technical Skills Demonstrated

- **Data Science**: Exploratory data analysis, statistical analysis, data visualization
- **Natural Language Processing**: Text preprocessing, tokenization, part-of-speech tagging, feature engineering
- **Linear Algebra**: Vector operations, matrix manipulations, dimensionality reduction
- **Machine Learning**: Unsupervised learning, clustering, similarity metrics, matrix factorization
- **Deep Learning**: Neural network architecture design, embedding techniques, model training and evaluation
- **Software Engineering**: Data transformation pipelines, algorithm implementation, performance evaluation

## Project Structure

The implementation follows a logical progression through nine lab notebooks:
1. **Exploratory Data Analysis**: Analyzes course popularity and user engagement patterns
2. **Bag of Words**: Extracts features from course textual content
3. **Similarity Calculation**: Implements methods to measure course similarity
4. **Content-Based Recommendation (Genre)**: Builds a recommender using course genres
5. **Content-Based Recommendation (Similarity)**: Creates recommendations based on course similarity
6. **User Clustering**: Groups users with similar interests for community-based recommendations
7. **Collaborative KNN**: Implements memory-based collaborative filtering
8. **Collaborative NMF**: Implements model-based collaborative filtering with matrix factorization
9. **Neural Network Approach**: Develops a deep learning model for rating prediction

## Key Results

- Content-based methods successfully identify courses with similar content profiles
- User clustering effectively groups learners with similar interests into distinct learning communities
- Matrix factorization techniques provide scalable solutions that address limitations of memory-based approaches
- Neural networks simultaneously learn compact representations while predicting ratings, offering advantages in both computational efficiency and predictive accuracy

## Dependencies

The implementation relies on several key libraries:
- pandas, numpy, matplotlib, seaborn for data manipulation and visualization
- nltk, gensim for natural language processing
- scikit-learn for machine learning algorithms
- surprise for collaborative filtering
- tensorflow/keras for neural network development

This project was completed as a Capstone as part of the IBM Machine Learning Professional Certificate.