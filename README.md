# Cancer-cell-classification-using-Scikit-learn
 
# Cancer Cell Classification using Scikit-learn

## Table of Contents
- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Development and Evaluation](#model-development-and-evaluation)
- [Conclusion](#conclusion)

## Introduction

  In this project, we'll classify cancer cells as 'malignant' or 'benign' based on their features using the Scikit-learn library for machine learning. Scikit-learn is an open-source library for machine learning, data mining, and data analysis in Python.
Machine Learning is a sub-field of Artificial Intelligence that gives systems the ability to learn themselves without being explicitly programmed to do so. Machine Learning can be used in solving many real world problems. 
Let’s classify cancer cells based on their features, and identifying them if they are ‘malignant’ or ‘benign’. We will be using scikit-learn for a machine learning problem. Scikit-learn is an open-source machine learning, data mining and data analysis library for Python programming language.
The dataset: 
Scikit-learn comes with a few small standard datasets that do not require downloading any file from any external website. The dataset that we will be using for our machine learning problem is the Breast cancer wisconsin (diagnostic) dataset. The dataset includes several data about the breast cancer tumors along with the classifications labels, viz., malignant or benign. It can be loaded using the following function: 
 

load_breast_cancer([return_X_y])
The data set has 569 instances or data of 569 tumors and includes data on 30 attributes or features like the radius, texture, perimeter, area, etc. of a tumor. We will be using these features to train our model.
Installing the necessary modules: 
For this machine learning project, we will be needing the ‘Scikit-learn’ Python module.
### The Dataset

We'll use the Breast cancer Wisconsin (diagnostic) dataset from Scikit-learn, which contains data on various attributes and the classification labels of cancer cells as either malignant or benign. This dataset comes pre-loaded with Scikit-learn.

To load the dataset, we'll use the `load_breast_cancer` function.

## Importing Libraries

First, we need to import the necessary Python modules and libraries, including Scikit-learn.

```python
import sklearn
from sklearn.datasets import load_breast_cancer
```

## Importing Dataset

Load the Breast cancer Wisconsin dataset using the `load_breast_cancer` function.

```python
data = load_breast_cancer()
```

## Exploratory Data Analysis (EDA)

Before building the model, let's explore and understand the dataset. We'll look at label names, labels, feature names, and features.

```python
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
```

## Feature Engineering

The dataset contains attributes like 'mean radius,' 'mean texture,' 'mean perimeter,' and more. These features will be used to train our model. The data should be preprocessed to make it suitable for training.

## Model Development and Evaluation

We will build a machine learning model using the Naive Bayes algorithm for binary classification. After training the model, we'll make predictions on a test set. We'll evaluate the model's accuracy using the `accuracy_score` function from Scikit-learn.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
train, test, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=42)

# Initialize and train the classifier
gnb = GaussianNB()
model = gnb.fit(train, train_labels)

# Make predictions
predictions = gnb.predict(test)

# Evaluate the model's accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
```

## Conclusion

In this project, we successfully built a machine learning model to classify cancer cells as malignant or benign based on their features. We used Scikit-learn and the Naive Bayes algorithm for classification. The model achieved an accuracy of approximately 94.15%.

 
