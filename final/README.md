# Spotify Song Popularity Prediction

This project aims to build a machine learning model that predicts the popularity score (0 to 100) of songs on Spotify based on their audio features.

## Data Source

https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db

## Project Structure
1.	Data Loading & Preprocessing
2.	Exploratory Data Analysis
3.	Model Definition 
4.	Model Training
5.	Evaluation using Regression Metrics
6.	Model Comparison & Final Selection

## Observations

- TabNet achieved the best performance in terms of MSE and RMSE.
- SimpleNN, DropoutNN, and MLP were very close in performance, with faster training time and simplicity.
- Linear Regression underperformed, confirming the problem is non-linear.
- Top 6 models performed very similarly, suggesting the dataset is well-suited for multiple architectures.

## Technologies Used
- pandas
- numpy
- scikit-learn
- torch
- pytorch
- matplotlib

## Future Work

- Add more audio or metadata features (lyrics, genres, artist popularity)
- Try additional regularization or advanced architectures
- Deploy the model in a web app 
