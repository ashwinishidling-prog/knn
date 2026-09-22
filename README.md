# Iris Species Classification using K-Nearest Neighbors (KNN)

An end-to-end Machine Learning pipeline utilizing the **K-Nearest Neighbors (KNN)** algorithm to classify Iris flower species based on morphological measurements.

---

## 📌 Project Overview

This repository demonstrates multi-class classification on the classic **Iris Dataset** using `scikit-learn`. The primary goal is to predict the species of an Iris flower based on sepal and petal parameters.

Key Characteristics of KNN:
* **Algorithm Type:** Instance-based / Lazy Learning (No explicit training phase; model memorizes training instances).
* **Decision Rule:** Predicts class membership via majority voting among the $k$ nearest data points.
* **Distance Metric:** Calculates distance (e.g., Euclidean distance) between test samples and stored training samples.

---

## 📊 Dataset Summary

The **Iris Dataset** contains 150 instances evenly distributed across 3 species:

* **Classes:**
  1. `Setosa` (Label 0)
  2. `Versicolor` (Label 1)
  3. `Virginica` (Label 2)
* **Features:**
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)

---

## 🛠️ Pipeline Architecture

1. **Data Loading:** Load Iris dataset from `sklearn.datasets`.
2. **Feature Engineering:** Extract feature values into a structured `pandas.DataFrame`.
3. **Data Splitting:** Divide data into training (80%) and testing (20%) sets using `train_test_split`.
4. **Model Initialization:** Instantiate `KNeighborsClassifier` with hyperparameter $k = 3$ (odd $k$ selected to prevent voting ties).
5. **Model Fitting:** Store training vectors for neighbor computations.
6. **Inference & Evaluation:** Predict test sample classes and compute accuracy score.

---

## 🚀 Quickstart & Usage

### 1. Prerequisites

Ensure you have Python 3.x installed along with the required libraries:

```bash
pip install pandas matplotlib scikit-learn
