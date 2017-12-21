import glob
import csv
import os
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

num_classes = 20
sample_history = 10

def load_data():
    path = "./ml_data/*.csv"
    data = []
    labels = []

    print("loading data...")

    for fname in glob.glob(path):
        with open(fname, 'r') as infh:
            reader = csv.reader(infh, delimiter=';')

            for row in reader:
                r = np.array(row, dtype = float)
                rr = []
                for i in range(sample_history):
                     rr.append(r[i*7+1])
                #print(rr)
                data.append(rr)
                labels.append(r[-1])
 
    data = np.array(data)
    labels = np.array(labels)
    n = int(float(data.shape[0]) * 0.8)
    train_data = data[:n]
    train_labels = labels[:n]
    test_data = data[n:]
    test_labels = labels[n:]
    print("finished loading data")
    return train_data, train_labels, test_data, test_labels

def grid_search(param_grid, clf, train_data, train_labels):
    grid = GridSearchCV(clf, param_grid=param_grid)
    grid.fit(train_data, train_labels)
    print("Best: {0}".format(grid.best_estimator_))
    return grid.best_estimator_

def plot_confusion_matrix(con_mat, nor = False):
    if nor:
        con_mat = con_mat.astype(float) / con_mat.sum(axis=0)
    plt.imshow(con_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(num_classes), np.arange(num_classes))
    plt.yticks(np.arange(num_classes), np.arange(num_classes))
    fmt = '.2f' if nor else '.0f'
    threshhold = con_mat.max() / 2.
    for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
        plt.text(j, i, format(con_mat[i, j], fmt),
             horizontalalignment="center",
             color="white" if con_mat[i, j] > threshhold else "black")
    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')


def train(train_data, train_labels, test_data, test_labels):
    param_grid = {
        "alpha":[0.0001, 0.01, 0.1, 1]
    }
    clf = MLPClassifier()
    clf = grid_search(param_grid, clf, train_data, train_labels)
    print("Train accuracy = {}".format(clf.score(train_data, train_labels)))
    print("Test accuracy = {}".format(clf.score(test_data, test_labels)))

    #confusion matrix
    test_pred = clf.predict(test_data)
    con_mat = np.zeros((num_classes, num_classes))
    for t, l in zip(test_pred, test_labels):
        con_mat[int(t)][int(l)] += 1

    print("Confusion Matrix = {}".format(con_mat))
    plt.figure()
    plot_confusion_matrix(con_mat, nor = True)
    plt.show()



if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels = load_data()
    train(train_data, train_labels, test_data, test_labels)


