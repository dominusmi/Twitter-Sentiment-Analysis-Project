#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import testsets
import evaluation
import pickle
from Classifier1 import Classifier1
from Classifier2 import Classifier2
from Classifier3 import Classifier3

'''
    IMPORTANT NOTE:
    The paths to the semval datasets are assumed to be in the same category, if this is
    not the case, please uncomment line 37 and change the paths_same_folder arguments to True

    Also, the first two classifiers are pretty fast, but the third takes about 7 minutes on my personal laptop,
    please don't be mad poor classifier has to deal with many dimensions
'''


for classifier in ['Lexicon-Classifier', 'ANN', 'Full Embedding']: # You may rename the names of the classifiers to something more descriptive
    clf = None
    if classifier == 'Lexicon-Classifier':
        print('Training ' + classifier)
        clf = Classifier1(loading=True, paths_same_folder=False)

    elif classifier == 'ANN':
        print('Training ' + classifier)
        clf = Classifier2(loading=True, paths_same_folder=False)

    elif classifier == 'Full Embedding':

        print('Training ' + classifier)
        clf = Classifier3(loading=True, paths_same_folder=False)

    for testset in testsets.testsets:

        # If testsets are in a sub category
        testset = "semeval-tweets/" + testset

        clf.load_test(testset)
        predictions = clf.predict()

        # predictions = {'163361196206957578': 'neutral', '768006053969268950': 'neutral', '742616104384772304': 'neutral', '102313285628711403': 'neutral', '653274888624828198': 'neutral'} # TODO: Remove this line, 'predictions' should be populated with the outputs of your classifier
        evaluation.evaluate(predictions, testset, classifier)

        evaluation.confusion(predictions, testset, classifier)
