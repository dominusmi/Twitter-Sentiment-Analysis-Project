import pickle
import numpy as np
import preprocessing as pp
import vocabulary_embedding as ve
import word_embedding as we
import lexicon_processing as lp

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import BaggingClassifier


class Classifier3:
    we_model = None

    def __init__(self, loading=True, paths_same_folder=False):
        self.loading = loading

        if paths_same_folder :
            train_paths = ["twitter-training-data.txt", "twitter-dev-data.txt"]
        else:
            train_paths = ["semeval-tweets/twitter-training-data.txt", "semeval-tweets/twitter-dev-data.txt"]

        # Loads initial preprocessed points
        tweets, unigram_stats = self.preprocessing(loading=loading, paths=train_paths)

        # Used in lexicon preparation if loading set to false
        lp.unigram_stats = unigram_stats
        lp.emojis_dict = lp.load_emoji_sentiment_dict()

        # Prepares all the various features to be put together
        print("Preparing lexicons")
        ids, Xl_train, y_train = self.preparing_lexicon(loading=loading, tweets=tweets)
        print("Preparing vocabulary embedding")
        Xv_train, yv_train = self.preparing_vocabulary_embedding(tweets)
        print("Preparing word embedding")
        Xw_train, yw_train = self.preparing_word_embedding(tweets)

        # Puts all features in one big matrix
        X_train = np.hstack( (Xl_train, Xw_train, Xv_train.toarray()) )

        self.estimators = [
            PassiveAggressiveClassifier(C=0.0001, loss='hinge', max_iter=100),
            RidgeClassifier(solver='auto', alpha=1000),
            BaggingClassifier(base_estimator=RidgeClassifier(), n_estimators=5, max_features=1.0, max_samples=0.2),
            BaggingClassifier(base_estimator=PassiveAggressiveClassifier(C=0.0001, max_iter=50),
                                n_estimators=8, max_features=1.0, max_samples=0.4),
            SGDClassifier(alpha=1e-2, max_iter=100)
        ]

        print("Fitting estimators")
        for clf in self.estimators:
            clf.fit(X_train, y_train)

    def load_test(self, path):
        test_tweets, _tweets_raw, _unigram_stats, _list = pp.load_dataset(path)

        self.ids_test, Xl_test, _y = self.preparing_lexicon(loading=False, tweets=test_tweets)
        Xv_test = self.transform_test_vocabulary_embedding(test_tweets)
        Xw_test, _y = self.preparing_word_embedding(test_tweets)

        self.X_test = np.hstack( (Xl_test, Xw_test, Xv_test.toarray()) )

    @staticmethod
    def democracy(votes):
        neg = np.sum( np.where(votes == -1) )
        neu = np.sum( np.where(votes == 0) )
        pos = np.sum( np.where(votes == 1) )

        if neg > neu and neg > pos:
            return -1
        elif neu > neg and neu > pos:
            return 0
        else:
            return 1

    def predict(self):

        results_dict = {}
        predictions = []

        for clf in self.estimators:
            pred = clf.predict(self.X_test)
            predictions.append( pred )

        predictions_grouped = np.array(predictions)

        democratic_vote = []
        for j in range(len(self.X_test)):
            democratic_vote.append( Classifier3.democracy( predictions_grouped[:,j] ) )

        for i, democratic_vote in enumerate(democratic_vote):
            results_dict[self.ids_test[i]] = pp.sentiment_from_code( democratic_vote )

        return results_dict

    '''
        Preprocesses data
        @param
            loading: whether to actually preprocess or load already processed
            paths: where to get variables
        @return
            preprocessed tweets and unigram stats
    '''
    def preprocessing(self, loading=True, paths=None):
        if loading:
            # Loads preprocessed training tweets
            with open("pickle/tweets_pp.p", "rb") as f:
                tweets = pickle.load(f)

            with open("pickle/unigram_stats.p", "rb") as f:
                unigram_stats = pickle.load(f)

        else:
            # Preprocesses training tweets
            tweets, tweets_raw, unigram_stats, _list = pp.load_dataset(paths)

            # Balances number of tweets
            count_negatives = np.sum( [1 for item in tweets.items() if item[1]['sentiment'] == -1] )
            tweets = pp.balance_samples(tweets, count_negatives)

        return tweets, unigram_stats

    '''
        Loads lexicons and ids from tweets list
        @param
            loading: whether to load or calculate variables
            tweets: tweets dictionary
        @return
            ids of tweets
            lexicon-vectorised tweets
            targets (not used here)
    '''
    def preparing_lexicon(self, loading=True, tweets=None):
        if loading:
            ids, Xl, yl, _headers = lp.load_training_lexicons()
        else:
            Xl, yl, _headers = lp.vectorise_tweets(tweets)
            ids = Xl[:,0]
            Xl = Xl[:,1:].astype('float')

        return ids, Xl, yl

    '''
        Applise an hashvectorizer to tweets and then chi2 squared tranformation choosing
        best 500
        @param
            list of test tweets
        @return
            Vocabulary embedding sparse matrix
            targets (not used in actual classification)
    '''
    def preparing_vocabulary_embedding(self, tweets):
        Xv, yv = ve.lemmas_to_strings( tweets )

        Xv, self.vectorizer = ve.hashvectorizer(Xv, ngram_range=(1,3))

        self.ch2_vocabulary = SelectKBest(chi2, k=500)
        Xv = self.ch2_vocabulary.fit_transform( Xv, yv )

        return Xv, yv

    '''
        Applies the transformations found from training data
        @param
            list of test tweets
        @return
            Vocabulary embedding sparse matrix
    '''
    def transform_test_vocabulary_embedding(self, test_tweets):
        Xv_test, _yv_test = ve.lemmas_to_strings( test_tweets )
        Xv_test = self.vectorizer.transform(Xv_test)
        Xv_test = self.ch2_vocabulary.transform(Xv_test)

        return Xv_test

    '''
        Loads the word embedding model and embeds the tweets
        @param
            tweets: list of tweets
        @return
            Wrod embedded tweets
    '''
    def preparing_word_embedding(self, tweets):
        if self.we_model is None:
            self.we_model = we.loadGloveModel("external-datasets/glove.twitter.27B/glove.twitter.27B.200d.txt")
            self.model_d = 200
            we.model = self.we_model
            we.model_d = self.model_d

        Xw, yw = ve.lemmas_to_strings( tweets )
        Xw = we.embed_tweets(Xw)

        return Xw, yw
