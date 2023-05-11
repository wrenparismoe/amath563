import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import PCA from sklearn
from sklearn.decomposition import PCA


train = pd.read_csv('MNIST/mnist_train.csv')
test = pd.read_csv('MNIST/mnist_test.csv')

# Split the data into X and y
x_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values

x_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values


# extract just the 3s and 8s
def extract_numbers(x_train, y_train, x_test, y_test, n, m):
    x_train = x_train[(y_train == n) | (y_train == m)]
    y_train = y_train[(y_train == n) | (y_train == m)]
    x_test = x_test[(y_test == n) | (y_test == m)]
    y_test = y_test[(y_test == n) | (y_test == m)]
    return x_train, y_train, x_test, y_test

def get_pca_modes(x_train):
    cov = np.cov(x_train.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = np.real(eigenvectors[:,idx])

    # Compute the cumulative sum of the sorted eigenvalues
    total = sum(eigenvalues)
    var_exp = [(i / total) for i in sorted(eigenvalues, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)

    # Determine the number of modes needed to preserve 95% of the variance
    k = np.argmax(cum_var_exp >= 0.95) + 1
    print("Number of modes to preserve 95% of variance:", k)
    return k

n = 5
m = 2
x_train, y_train, x_test, y_test = extract_numbers(x_train, y_train, x_test, y_test, n, m)

# Create a PCA object
pca = PCA()

# print("pre-pca ckpt")
# Fit the PCA object to the training data
k = get_pca_modes(x_train)
pca.fit(x_train)

# print("pre-pca transform ckpt")
# Transform the training and test data using the PCA object
x_train_pca = pca.transform(x_train)
x_test_pca = pca.transform(x_test)

# Cast labels from 1,9 to 0,1
y_train = np.where(y_train == n, 0, 1)
y_test = np.where(y_test == n, 0, 1)

# Perform kernel regression to classify the data with a linear, polynomial, and RBF kernel
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

print("Running kernel regression")
# Create a linear SVM classifier
linear_svm = SVC(kernel='linear')

# Train the classifier
linear_svm.fit(x_train_pca, y_train)

# Make predictions on the test data
y_pred = linear_svm.predict(x_test_pca)

# Compute the training and test accuracy
train_acc = accuracy_score(y_train, linear_svm.predict(x_train_pca))
test_acc = accuracy_score(y_test, y_pred)

print("Linear Kernel: Training Accuracy = %f, Test Accuracy = %f" % (train_acc, test_acc))

# Create a polynomial SVM classifier
poly_svm = SVC(kernel='poly', degree=2)

# Train the classifier
poly_svm.fit(x_train_pca, y_train)

# Make predictions on the test data
y_pred = poly_svm.predict(x_test_pca)

# Compute the training and test accuracy
train_acc = accuracy_score(y_train, poly_svm.predict(x_train_pca))
test_acc = accuracy_score(y_test, y_pred)

print("Polynomial Kernel: Training Accuracy = %f, Test Accuracy = %f" % (train_acc, test_acc))

# Create a RBF SVM classifier
rbf_svm = SVC(kernel='rbf')

# Train the classifier
rbf_svm.fit(x_train_pca, y_train)

# Make predictions on the test data
y_pred = rbf_svm.predict(x_test_pca)

# Compute the training and test accuracy
train_acc = accuracy_score(y_train, rbf_svm.predict(x_train_pca))
test_acc = accuracy_score(y_test, y_pred)

print("RBF Kernel: Training Accuracy = %f, Test Accuracy = %f" % (train_acc, test_acc))




