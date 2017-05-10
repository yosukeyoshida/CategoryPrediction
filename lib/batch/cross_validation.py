import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../config')
import config

from sklearn import cross_validation
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from datetime import datetime
import pdb
import numpy as np

from document_reader import DocumentReader
from preprocessor import Preprocessor

def get_accuracy(clf, train_features, train_labels):
    scores = cross_validation.cross_val_score(clf, train_features, train_labels, cv=10)
    print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

if __name__ == "__main__":
    data, label, ids = DocumentReader.load_texts()

    penalty = config.MODEL_OPTIONS["penalty"]
    C = config.MODEL_OPTIONS["C"]
    estimator = LogisticRegression(penalty=penalty, C=C)

    TYPE = 1
    # 1: cross validation  2: learning curve
    if TYPE == 1:
        k_fold = cross_validation.KFold(n=len(data), n_folds = 10, shuffle=True)

        scores = []
        for train_index, test_index in k_fold:
            X_train = data[train_index]
            y_train = label[train_index]
            X_test = data[test_index]
            y_test = label[test_index]

            preprocessor = Preprocessor(X_train)

            X_train_lsi = preprocessor.data2dense(X_train)
            X_test_lsi = preprocessor.data2dense(X_test)

            estimator.fit(X_train_lsi, y_train)
            score = estimator.score(X_test_lsi, y_test)
            print(score)
            scores.append(score)

        print(scores)
        mean = np.mean(scores)
        std = np.std(scores) * 2
        print("総件数: %s" % len(data))
        print("Mean: %f" % mean)
        print("Std * 2: %f" % std)

        preprocessor2 = Preprocessor(data)
        data_lsi = preprocessor2.data2dense(data)
        estimator.fit(data_lsi, label)
        print(estimator.score(data_lsi, label))
    elif TYPE == 2:
        preprocessor = Preprocessor(data)
        data_lsi = preprocessor.data2dense(data)
        training_sizes, train_scores, test_scores = learning_curve(estimator=estimator, X=data_lsi, y=label, train_sizes=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], cv=10)

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.plot(training_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
        plt.fill_between(training_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

        plt.plot(training_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
        plt.fill_between(training_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

        plt.grid()
        plt.xlabel('Number of training samples')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.ylim([0.5, 1.0])
        plt.show()