# Importing necessary libraries
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load datasets
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

cancer = load_breast_cancer()
X_cancer, y_cancer = cancer.data, cancer.target

def evaluate_classifiers(X, y, classifiers, train_size=0.75):
    """
    Evaluate classifiers on given dataset using specified training size.
    
    Parameters:
        X: Features
        y: Labels
        classifiers: Dictionary containing classifiers
        train_size: Size of training set
        
    Returns:
        Dictionary containing accuracy scores for each classifier
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    
    for name, clf in classifiers.items():
        clf.fit(X_train_scaled, y_train)
        score = clf.score(X_test_scaled, y_test)
        results[name] = score
    
    return results

def evaluate_cross_validation(X, y, classifiers, cv=5):
    """
    Evaluate classifiers on given dataset using cross-validation.
    
    Parameters:
        X: Features
        y: Labels
        classifiers: Dictionary containing classifiers
        cv: Number of folds for cross-validation
        
    Returns:
        Dictionary containing mean accuracy scores for each classifier
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = {}
    
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv)
        mean_score = np.mean(scores)
        results[name] = mean_score
    
    return results

# Define classifiers
classifiers = {
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Evaluate classifiers on Iris dataset
print("Iris Dataset:")
print("Training set = 75%, Test set = 25%")
results_iris_75 = evaluate_classifiers(X_iris, y_iris, classifiers)
print(results_iris_75)

print("Training set = 66.6% (2/3rd of total), Test set = 33.3%")
results_iris_66 = evaluate_classifiers(X_iris, y_iris, classifiers, train_size=2/3)
print(results_iris_66)

print("Training set chosen by Cross-Validation")
results_iris_cv = evaluate_cross_validation(X_iris, y_iris, classifiers)
print(results_iris_cv)

# Evaluate classifiers on Breast Cancer Wisconsin dataset
print("\nBreast Cancer Wisconsin Dataset:")
print("Training set = 75%, Test set = 25%")
results_cancer_75 = evaluate_classifiers(X_cancer, y_cancer, classifiers)
print(results_cancer_75)

print("Training set = 66.6% (2/3rd of total), Test set = 33.3%")
results_cancer_66 = evaluate_classifiers(X_cancer, y_cancer, classifiers, train_size=2/3)
print(results_cancer_66)

print("Training set chosen by Cross-Validation")
results_cancer_cv = evaluate_cross_validation(X_cancer, y_cancer, classifiers)
print(results_cancer_cv)
