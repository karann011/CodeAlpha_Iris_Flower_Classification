# Iris Flower Classification using Machine Learning

## Project Overview
This project classifies Iris flowers into three species: Setosa, Versicolor, and Virginica using Machine Learning.

## Objective
To build a classification model that predicts the species of an Iris flower based on its measurements.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

## Dataset
The Iris dataset is loaded using Scikit-learn. It contains four input features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target consists of three Iris flower species.

## Machine Learning Algorithm
Logistic Regression is used to train the classification model. StandardScaler is included in a pipeline to scale the input features.

## Project Workflow
1. Load the Iris dataset.
2. Prepare the input features and target labels.
3. Split the dataset into training and testing sets.
4. Train the Logistic Regression model.
5. Predict flower species using test data.
6. Evaluate model accuracy and generate a classification report.
7. Create and save a confusion matrix.
8. Predict the species of a new flower.

## How to Run
Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

## Output
The program displays the model accuracy, classification report, confusion matrix, and prediction for a sample flower.

## Conclusion
This project demonstrates a basic Machine Learning classification workflow using Python and Scikit-learn.
