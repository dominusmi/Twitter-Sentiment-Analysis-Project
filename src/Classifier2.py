import numpy as np
import pickle
import keras

import preprocessing as pp
import lexicon_processing as lp

class Classifier2:
    X_test = None
    ids_test = None
    ANN = None

    def __init__(self, loading=True, paths_same_folder=False):
        if paths_same_folder :
            train_paths = ["twitter-training-data.txt", "twitter-dev-data.txt"]
        else:
            train_paths = ["semeval-tweets/twitter-training-data.txt", "semeval-tweets/twitter-dev-data.txt"]


        if loading:
            with open("pickle/unigram_stats.p", "rb") as f:
                unigram_stats = pickle.load(f)

        else:
            # Only need unigram_stats for this model
            tweets, tweets_raw, unigram_stats, _list = pp.load_dataset(train_paths)

        lp.unigram_stats = unigram_stats
        lp.emojis_dict = lp.load_emoji_sentiment_dict()

        self.ANN = keras.models.load_model("models/ANN7")

    @staticmethod
    def categorical_to_int(predictions):
        # Required to switch back from categorical
        predictions[ predictions == 2 ] = -1

        return predictions

    def load_test(self, path):
        # Loads testset
        tweets_test, _raw, _ugs, _list = pp.load_dataset(path)
        x, y , _headers = lp.vectorise_tweets(tweets_test)

        self.ids_test = x[:,0]
        self.X_test = x[:,1:].astype('float')

    def predict(self):
        results_dict = {}

        predictions = self.ANN.predict(self.X_test)
        predictions = np.argmax(predictions, axis=1)
        predictions = Classifier2.categorical_to_int(predictions)

        for i, predictions in enumerate(predictions):
            results_dict[self.ids_test[i]] = pp.sentiment_from_code( predictions )

        return results_dict
