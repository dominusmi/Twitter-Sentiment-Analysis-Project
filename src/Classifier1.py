import preprocessing as pp
import testsets as ts
import lexicon_processing as lp
import pickle
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import BaggingClassifier


class Classifier1:
    ch2 = None
    estimators = None
    X_test = None
    ids_test = None

    def __init__(self, loading=True, paths_same_folder=False):
        if paths_same_folder :
            train_paths = ["twitter-training-data.txt", "twitter-dev-data.txt"]
        else:
            train_paths = ["semeval-tweets/twitter-training-data.txt", "semeval-tweets/twitter-dev-data.txt"]
            
        if loading:
            # Loads preprocessed training tweets
            with open("pickle/tweets_pp.p", "rb") as f:
                tweets = pickle.load(f)

            with open("pickle/unigram_stats.p", "rb") as f:
                unigram_stats = pickle.load(f)

        else:
            # Preprocesses training tweets
            tweets, tweets_raw, unigram_stats, _list = pp.load_dataset(train_paths)

            # Balances number of tweets
            count_negatives = np.sum( [1 for item in dataset.items() if item[1]['sentiment'] == -1] )
            tweets = pp.balance_samples(tweets, count_negatives)

        lp.unigram_stats = unigram_stats
        lp.emojis_dict = lp.load_emoji_sentiment_dict()


        if loading:
            # Loads vectorised tweets
            ids, X_train, y_train, headers = lp.load_training_lexicons()

        else:
            # Generates vectorised tweets
            X_train, y_train, headers = lp.vectorise_tweets(tweets)
            X_train = X_train[:,1:].astype('float')


        # Features selection using chi squared test
        self.ch2 = SelectKBest(chi2, k=10)
        X_train = self.ch2.fit_transform(X_train, y_train)

        # Train classifiers
        self.estimators = [
            PassiveAggressiveClassifier(C=0.0001, loss='squared_hinge', max_iter=500),
            RidgeClassifier(solver='auto', alpha=1000),
            BaggingClassifier(base_estimator=RidgeClassifier(), n_estimators=5, max_features=1.0, max_samples=0.2)
        ]

        for clf in self.estimators:
            clf.fit(X_train, y_train)


    def load_test(self, path):
        # Loads testset
        tweets_test, _raw, _ugs, _list = pp.load_dataset(path)

        self.X_test, y_test, ids_test = None, None, None
        x, y , _headers = lp.vectorise_tweets(tweets_test)

        self.ids_test = x[:,0]
        y_test = y

        self.X_test = x[:,1:].astype('float')
        self.X_test = self.ch2.transform(self.X_test)


    def predict(self):
        results_dict = {}

        results = lp.predict_democratic(self.estimators, self.X_test)

        for i, result in enumerate(results):
            results_dict[self.ids_test[i]] = pp.sentiment_from_code( result )

        return results_dict
