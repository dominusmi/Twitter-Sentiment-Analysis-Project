import pickle
from time import time
import matplotlib.pyplot as plt
import matplotlib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.feature_selection import SelectFromModel
import numpy as np

# Benchmark classifiers
def benchmark_f1(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    scores = []
    for i in range(3):
        t0 = time()
        pred = clf.predict(X_test[i])
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test[i], pred)
        print("accuracy:   %0.3f" % score)

        scores.append( metrics.f1_score(y_test[i], pred, average='weighted'))


    clf_descr = str(clf).split('(')[0]
    return clf_descr, scores, train_time, test_time

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

#         if opts.print_top10 and feature_names is not None:
#             print("top 10 keywords per class:")
#             for i, label in enumerate(target_names):
#                 top10 = np.argsort(clf.coef_[i])[-10:]
#                 print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
#         print()

#     if opts.print_report:
    print("classification report:")
    print(metrics.classification_report(y_test, pred))

#     if opts.print_cm:
    print("confusion matrix:")
    print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time
