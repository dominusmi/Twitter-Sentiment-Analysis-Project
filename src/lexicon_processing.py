import pickle
import numpy as np

# Needs to be set after preprocessing
unigram_stats = None
emojis_dict = None


'''
    Loads processed lexicons of training tweets
    @return
        list of tweet ids
        list of training tweets
        lsit of training targets
'''
def load_training_lexicons(scale=False):
    with open("pickle/vectorised_train.p", "rb") as f:
        train_dict = pickle.load(f)
        train_tweets = train_dict.get('vectorised')
        train_targets = train_dict.get('targets')
        headers = train_dict.get('headers')


    train_ids = train_tweets[:,0]
    train_tweets = train_tweets[:,1:].astype('float')

    if scale:
        train_tweets = np.scale(train_tweets)

    return  train_ids, train_tweets, train_targets, headers

'''
    Loads processed lexicons of training tweets
    @return
        list of tweet ids
        list of testing tweets
        list of testing targets
'''
def load_testing_lexicons(scale=False):
    test_tweets = []
    test_targets = []
    test_ids = []
    for i in [0,1,2]:
        with open("pickle/vectorised_test{}.p".format(i), "rb") as f:
            test_dict = pickle.load(f)

            _ids = test_dict.get('vectorised')[:,0]
            _tweets = test_dict.get('vectorised')[:,1:].astype('float')

            if scale:
                _tweets = np.scale(_tweets)

            test_ids.append(_ids)
            test_tweets.append( _tweets )
            test_targets.append( test_dict.get('targets') )



    return test_ids, test_tweets, test_targets


'''
    Loads dictionary containing emojis and their probability of being associated
    with a given sentiment
    @return
        dictionary of emojis containg probabilities
'''
def load_emoji_sentiment_dict():
    with open("pickle/emoticon_labels.p", "rb") as f:
        emojis_dict = pickle.load(f)

    return emojis_dict


'''
    Acts similarly for emojis to what word synsets is for words, gives "probabilities"
    of emoji being associated with a sentiment
    @param
        tweet to get emoji labels from
    @return
        list of three entries, one for each sentiment, containing the sum of "probabilities"
'''
def get_emoticon_label(tweet):
    label = [0,0,0] #neg, neu, pos
    emojis = tweet.get('emojis')
    if len(emojis)>0:
        for emoji in emojis:
            emoji_dict = emojis_dict.get(emoji)
            if emoji_dict is not None:
                label[0] += emoji_dict.get(-1)
                label[1] += emoji_dict.get(0)
                label[2] += emoji_dict.get(1)

    return label


'''
    For each unigram, returns the probability of it being associated to a sentiment
    based on the training set
    @param
        tweet text
    @return
        list of three entries, one for each sentiment, with the sum of all the "probabilities"
'''
def get_unigram_sentiment(tweet):
    sentiment = {-1:0, 1:0, 0:0}
    for lemma in tweet:
        lemma_stats = unigram_stats.get(lemma)
        if lemma_stats is not None:
            sentiment[0] += lemma_stats[0]
            sentiment[1] += lemma_stats[1]
            sentiment[-1] += lemma_stats[-1]

    return [sentiment[-1], sentiment[0], sentiment[1]]


'''
    Creates a vector out of all the lexicon features extracted
    @param
        tweets: dictionary containing all features for each tweet
    @return
        np array of vectorised tweets
        np array of targets
        list of headers for each column
'''
def vectorise_tweets(tweets):
    pp_tweets = []
    targets = []
    for _id, tweet in tweets.items():
        pp_tweet = []

        targets.append(tweet.get('sentiment'))

        pp_tweet.append(_id)
        pp_tweet.append(tweet.get('length'))
        pp_tweet += get_emoticon_label(tweet)
        pp_tweet.append(len(tweet.get('hashtags')))
        pp_tweet.append(tweet.get('punctuation'))
        pp_tweet.append(tweet.get('exclamation'))
        pp_tweet.append(tweet.get('interrogation'))
        pp_tweet.append(tweet.get('dotdotdot'))
        pp_tweet.append(tweet.get('people'))
        pp_tweet.append(tweet.get('positive'))
        pp_tweet.append(tweet.get('negative'))
        pp_tweet.append(tweet.get('url'))
        # !!! Requires unigram_stats as global variable !!!
        pp_tweet += get_unigram_sentiment(tweet.get('tweet'))

        pp_tweet.append(tweet.get('negations'))
        pp_tweet.append(tweet.get('laughter'))
        pp_tweet.append(tweet.get('bad'))
        pp_tweet.append(tweet.get('religious'))
        pp_tweet.append(tweet.get('synset_pos'))
        pp_tweet.append(tweet.get('synset_neg'))


        pp_tweets.append(pp_tweet)

    pp_headers = ['length', 'em_neg', 'em_neu', 'em_pos', '#', 'punctuation', 'exclamation', 'interrogation',
                     '...', 'people', 'positive', 'negative', 'url', 'neg_unigrams', 'neu_unigrams',
                  'pos_unigrams', 'negations', 'laughter', 'bad','religious', 'synset_pos', 'synset_neg']


    return np.array(pp_tweets), np.array(targets), pp_headers


'''
    Loads pre trained models, all used together to compose Classifier 1
    @return
        three sklearn models
'''
def load_estimators():
    with open("pickle/clf1", "rb") as f:
        clf_dict = pickle.load(f)

    return [clf_dict.get('clf1'), clf_dict.get('clf2'), clf_dict.get('clf3')]

'''
    Given a vector of votes with values in (-1,0,1), gives the majority vote
    @param
        votes: vector of votes
    @return
        majority vote
'''
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


'''
    Predicts class on a dataset
    @params
        estimators: list of the n estimators
        X_test:     single testing dataset
    @return
        array of (-1,0,1) giving the prediction
'''
def predict_democratic(estimators, X_test):
    # For each test set, calculate predictions
    predictions = []
    for clf in estimators:
        pred = clf.predict(X_test)
        predictions.append( pred )

    # Then transform to matrix
    predictions_grouped = np.array(predictions)

    # Finally, get the democratic vote
    final = []
    for j in range(len(X_test)):
        final.append( democracy( predictions_grouped[:,j] ) )

    return final
