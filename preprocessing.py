import re
import pickle
import random
import numpy as np
import csv

from nltk import wordnet, word_tokenize, WordNetLemmatizer, pos_tag
from nltk.corpus import sentiwordnet as swn
from tqdm import tqdm
import testsets


#################### GENERAL CONSTANTS USED THROUGHOUT FUNCTIONS ####################
### REGEX patterns ###
# General
pattern_numbers = re.compile(r'\b([0-9.]+)\b')
pattern_non_alpha = re.compile(r'([^a-z0-9\s\<\>]|[\s]{2,})')
pattern_urls = re.compile(r"((?:http(s)?://)?t.co/[a-z0-9A-Z]+)\b")
pattern_single = re.compile(r"(\b(?:[b-h]|[j-z]))\b") # Keeps "a" and "I" since they provide meaning

# Twitter specific
pattern_hashtag = re.compile(r'#([a-z0-9]+)(?:\b|$)')
pattern_people = re.compile(r'\B@([a-z0-9]+)(?:\b|$)')
pattern_unicode = re.compile(r'(\\u[0-9]+)')

# Punctuation specific
pattern_exclamation = re.compile(r'!')
pattern_interrogation = re.compile(r'\?')
pattern_punctuation = re.compile(r'[\.,\?!;:-_!\"]')
pattern_dotdotdot = re.compile(r'\.\.\.')

# Copied from https://stackoverflow.com/questions/10072744/remove-repeating-characters-from-words
pattern_elonganted = re.compile(r'(.)\1+')

# Load word datasets
positive_lex    = None
negative_lex    = None
bad_words       = None
religious_words = None
negation_words  = None
laughter_words  = set(["haha", "aha", "ahah", "hahaha", "ahahah", "hihi", "hah", "lol", "rofl", "lmao", "lmfao"])



#################### FUNCTIONS ####################

'''
    Returns the integer corresponding to the sentiment, +1, 0, -1 for respectively
    positive, neutral and negative tweet
    @param:
        sentimet: string which should either be "positive", "neutral" or "negative"
'''
def sentiment_code(sentiment):
    if sentiment[0] == 'p':
        return 1
    elif sentiment[2] == 'u':
        return 0
    else:
        return -1


'''
    Finds and replaces hashtags with "<hashtag>" identifier
    @param
        tweet: text of the tweet
    @return
        String with hashtags replaced
        List of hashtags replaced
'''
def find_and_replace_hastags(tweet):
    hashtags = pattern_hashtag.findall(tweet)
    new_string = pattern_hashtag.sub("<hashtag>",tweet)
    return new_string, hashtags

'''
    Finds and replaces user mentions with "<user>" identifier
    @param
        tweet: text of the tweet
    @return
        String with user mentions replaced
        List of users replaced
'''
def find_and_replace_people(tweet):
    people = len(pattern_people.findall(tweet))
    new_string = pattern_people.sub("<user>",tweet)
    return new_string, people

'''
    Finds and replaces emojis with "". Note: only handles "face" emojis
    @param
        tweet: text of the tweet
    @return
        String with emojis replaced
        List of users replaced
'''
def find_and_replace_emojis(tweet):
    emojis = []
    to_remove = []

    for letter in tweet:
        _ord = ord(letter)

        if _ord >=128512 and _ord <= 128591:
            emojis.append(_ord)
            to_remove.append(letter)

    return tweet.strip("".join(to_remove)), emojis

'''
    Returns counts of punctuations. Also separately returns counts of "?", "!", "..."
    @param
        tweet text
    @return
        count of all punctuations
        count of exclamation marks
        count of question marks
        count of "..."
'''
def count_punctuation(tweet):
    c_exclamation   = len(pattern_exclamation.findall(tweet))
    c_interrogation = len(pattern_interrogation.findall(tweet))
    c_punctuation   = len(pattern_punctuation.findall(tweet))
    c_dotdotdot     = len(pattern_dotdotdot.findall(tweet))

    return c_punctuation, c_exclamation, c_interrogation, c_dotdotdot

'''
    Other regex patterns which simply get rid of types of characters in tweets
'''
count_and_remove_urls   = lambda tweet: pattern_urls.subn("<url>",tweet)
remove_numbers          = lambda tweet: pattern_numbers.sub("", tweet)
remove_unicode          = lambda tweet: pattern_unicode.sub(" ", tweet)
remove_non_alphanumeric = lambda tweet: pattern_non_alpha.sub(" ", tweet)
remove_single           = lambda tweet: pattern_single.sub("", tweet)
remove_elongated        = lambda tweet: pattern_elonganted.sub(r'\1\1', tweet)


'''
    Returns tag used for lemmatizing tokens, given the pos_tag of the token
    @param
        pos_tag of token
    @return
        lemmatization compatible tag (Defaults to noun)
'''
def get_lemma_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return 'n'

'''
    Calculates base 10 entropy of a word based on the number of appearences in the
    types of tweets. For instance, a word like "shit" is largely used in negative tweets,
    and therefore carries more information than "the".

    @param
        Dictionary of word, containing its appearences in the three types of tweets,
        and its total appearences
    @return
        base 10 entropy
'''
def entropy(_ngram_dict):
    entropy = 0
    for i in [-1,0,1]:
        if _ngram_dict[i] == 0:
            continue
        p = _ngram_dict[i] / _ngram_dict['count']
        entropy += p * np.log10(p)
    return entropy

'''
    Restructures unigrams by filtering out words that either don't carry enough information.
    Computes the weighted probability for each of the remaining word to be in a
    positive, negative of neutral tweet

    @params
        unigrams:       Dictionary for each word of counts in types of tweets
        counts:         Total coutns of types of tweets
        threshold:      How many times a word must appear to be considered usable
        entropy_diff:   Entropy threshold to keep or remove word
'''
def restructure_unigrams(unigrams, counts, threshold = 5, entropy_diff = 0.04):

    tot_positive = counts[1]
    tot_negative = counts[-1]
    tot_neutral = counts[0]

    to_remove = set()

    for key in unigrams:
        details = unigrams.get(key)

        if details['count'] < threshold:
            # Remove words that appear too little to give weight
            to_remove.add(key)

        elif np.abs( entropy(details)-0.5 ) < entropy_diff:
            # Remove words that don't carry enough information
            to_remove.add(key)
        else:
            # Divide by respective sentiment count
            details[-1] /= tot_negative
            details[0] /= tot_neutral
            details[1] /= tot_positive

            # Normalising to unbiased 'probability'
            n = details[0]+details[-1]+details[1]
            details[0] /= n; details[1] /= n; details[-1] /= n;

    # Remove filtered keys
    for key in to_remove: unigrams.pop(key);

    return unigrams

'''
    Balances samples to have as many neutral and positive tweets as negative,
    which appear much less often. (In test set, ~8k negative for ~20k/16k neutral/positive )
'''
def balance_samples(tweets, n_neg):

    keys=list(tweets.keys())
    random.shuffle(keys)
    final_dict = {}

    count_neu, count_pos = 0,0


    for key in keys:
        if tweets.get(key)['sentiment'] == -1:
            final_dict[key] = tweets.get(key)
        elif tweets.get(key)['sentiment'] == 0 and count_neu < n_neg:
            final_dict[key] = tweets.get(key)
            count_neu += 1
        elif tweets.get(key)['sentiment'] == 1 and count_pos < n_neg:
            final_dict[key] = tweets.get(key)
            count_pos += 1

    return final_dict


def get_vocabulary(unigrams):
    for i, key in enumerate(unigrams):
        unigrams.get(key)['id'] = i




'''
    Loads positive and negative words, additionally removing non-alphanumeric and short words
    @return:
        set of positive words
        set of negative words
'''
def load_sentiment_words():

    non_alphanumeric = re.compile(r'[^a-zA-Z\d]')
    negative = []
    positive = []

    # These two file opening could've been made into a function, but was actually becoming messier

    with open("external-datasets/negative-words.txt","r", encoding='latin-1') as f:
        for line in f.readlines():
            if line[0] == ';':
                continue
            # Process the data on the go
            tmp = line.rstrip("\n")
            negative.append(tmp)

    with open("external-datasets/positive-words.txt","r", encoding='latin-1') as f:
        for line in f.readlines():
            if line[0] == ';':
                continue

            tmp = line.rstrip("\n")
            positive.append(tmp)

    return set(positive), set(negative)


'''
    Loads bad words, additionally removing non-alphanumeric and short words
    @return:
        set of bad words
'''
def load_bad_words():

    bad = []


    with open("external-datasets/bad-words.txt","r", encoding='latin-1') as f:
        for line in f.readlines():
            # Process the data on the go
            tmp = line.rstrip("\n")
            bad.append(tmp)

    return set(bad)

'''
    Load religious words
    @return
        set of religious words
'''
def load_religious_words():
    return set(["christian", "muslim", "islam", "jew", "coran", "bible", "quran", "hindu"])

'''
    Loads negation words
    @return
        set of negation words
'''
def load_negation_words():
    ngs = []

    with open("external-datasets/negations.txt","r", encoding='latin-1') as f:
        for line in f.readlines():
            # Process the data on the go
            tmp = line.rstrip("\n")
            ngs.append(tmp)

    return set(ngs)



'''
    Finds how many positive and negative words in an article
    @params:
        article:    List of lemmas constituing article
        positive:   set of positive words
        negative:   set of negative words
    @return:
        dictionary with 'positive' and 'negative' entries
'''
def compute_sentiment(article, positive, negative):
    pos, neg = 0,0
    for word in article:
        if word in negative:
            neg+=1
        elif word in positive:
            pos+=1
    return {'positive':pos,'negative':neg}

'''
    Counts number of words from given set
    @params:
        tweet:      lemmatized tweet
        words_set:  given set of words, could be positive, negative, religious..
    @return
        count of religious words

'''
def count_in_set(tweet, words_set, binary=False):
    count = 0
    for word in tweet:
        if word in words_set:
            count += 1

    if binary:
        return int(bool(count))
    else:
        return count


'''
    Balances samples by leaving as many positive and neutral tweets as there are negative.
    @params
        tweets: tweets dictionary
        n_neg: number of negative tweets
    @return
        tweets: tweets_dictionary with removed entries
'''
def balance_samples(tweets, n_neg):

    keys=list(tweets.keys())
    random.shuffle(keys)
    final_dict = {}

    count_neu, count_pos = 0,0


    for key in keys:
        if tweets.get(key)['sentiment'] == -1:
            final_dict[key] = tweets.get(key)
        elif tweets.get(key)['sentiment'] == 0 and count_neu < n_neg:
            final_dict[key] = tweets.get(key)
            count_neu += 1
        elif tweets.get(key)['sentiment'] == 1 and count_pos < n_neg:
            final_dict[key] = tweets.get(key)
            count_pos += 1

    return final_dict


'''
    Main loading function. Calls all the other functions and puts information together
    @param
        paths: either single path (str) or list of paths for which tweets will be concatenated
    @return
        dataset:        dictionary of tweets containing all features extracted
        unprocessed:    dictionary of unprocessed tweets, used for debug
        unigrams_stats: dictionary which contains all unigrams and their statistics
        tweets_list:    list of tweets used for debug
'''
def load_dataset( paths ):

    if type(paths) is str:
        paths = [paths]

    global positive_lex

    if positive_lex is None:
        global negative_lex, bad_words, religious_words, negation_words
        positive_lex, negative_lex = load_sentiment_words()
        bad_words                  = load_bad_words()
        religious_words            = load_religious_words()
        negation_words             = load_negation_words()

    sentiments_count = {1:0, -1:0, 0:0}

    dataset = {}
    unprocessed = {}
    unigrams_stats = {}

    tweets_list = []

    wnl = WordNetLemmatizer()

    for path in paths:
        file_length = sum(1 for line in open(path))

        with open(path) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in tqdm(reader, total=file_length):

                # Get integer for tweet sentiment (-1,0,1)
                tweet_sentiment = sentiment_code(row[1])

                # Update number of pos/neu/neg tweets
                sentiments_count[ tweet_sentiment ] += 1

                # Save raw data for debug
                unprocessed[row[0]] = [row[1], row[2]]

                # Process text, remove and count
                tweet = row[2].lower()
                tweet_length = len(tweet.split())
                tweet, c_url = count_and_remove_urls(tweet)
                tweet = remove_numbers(tweet)
                tweet, hashtags = find_and_replace_hastags(tweet)
                tweet, people = find_and_replace_people(tweet)
                tweet, emojis = find_and_replace_emojis(tweet)
                c_punct, c_excl, c_int, c_ddd = count_punctuation(tweet)
                tweet = remove_elongated(tweet)
                tweet = remove_non_alphanumeric(tweet)
                tweet = remove_single(tweet)

                tweets_list.append(tweet)

                # Tokenize
                tweet = word_tokenize(tweet)

                # Lemmatize
                tags = pos_tag(tweet)
                pos_tags = []
                for i,token in enumerate(tweet):
                    pos = get_lemma_pos(tags[i][1])
                    tweet[i] = wnl.lemmatize(tweet[i], pos=pos)
                    pos_tags.append(pos)


                # Synset score
                tweet_synset={'pos':0.0, 'neg':0.0}
                count_synset = 0
                for i, token in enumerate(tweet):

                    _synset = list(swn.senti_synsets(token, pos_tags[i]))
                    if len(_synset) > 0:
                        tweet_synset['pos'] += _synset[0].pos_score()
                        tweet_synset['neg'] += _synset[0].neg_score()
                        count_synset += 1
                if count_synset > 0:
                    tweet_synset['pos'] /= count_synset
                    tweet_synset['neg'] /= count_synset


                # Counts words in various already loaded sets
                t_raw = row[2].split(" ")
                pos_count = count_in_set(t_raw, positive_lex)
                neg_count = count_in_set(t_raw, negative_lex)
                rel_count = count_in_set(t_raw, religious_words, binary=True)
                bad_count = count_in_set(t_raw, bad_words)
                ngs_count = count_in_set(t_raw, negation_words)
                lgs_count = count_in_set(t_raw, laughter_words, binary=True)


                # Update vocabulary statistics
                for lemma in tweet:
                    stats = unigrams_stats.get(lemma)

                    if stats is None:
                        # Create array
                        unigrams_stats[lemma] = {1:0, 0:0, -1:0, 'count':0}
                        stats = unigrams_stats.get(lemma)

                    stats[tweet_sentiment]+=1
                    stats['count']+=1



                # Save in master dictionary
                dataset[row[0]] = {
                        'sentiment': tweet_sentiment,
                        'tweet': tweet,
                        'hashtags': hashtags,
                        'people': people,
                        'emojis':emojis,
                        'length': tweet_length,
                        'exclamation': c_excl,
                        'punctuation': c_punct,
                        'interrogation': c_int,
                        'dotdotdot': c_ddd,
                        'url': c_url,
                        'positive': pos_count,
                        'negative': neg_count,
                        'delta_emotion': pos_count - neg_count,
                        'bad': bad_count,
                        'religious': rel_count,
                        'negations': ngs_count,
                        'laughter': lgs_count,
                        'synset_pos': tweet_synset['pos'],
                        'synset_neg': tweet_synset['neg']
                    }


    unigrams_stats = restructure_unigrams(unigrams_stats, sentiments_count)

    return dataset, unprocessed, unigrams_stats, tweets_list


'''
    Loads preprocessed training tweets
    @return
        dictionary of tweets
'''
def load_training(balanced=True):
    if balanced:
        with open("pickle/tweets_pp.p", "rb") as f:
            tweets = pickle.load(f)
    else:
        with open("pickle/tweets_nobalance_pp.p", "rb") as f:
            tweets = pickle.load(f)
    return tweets
'''
    Loads preprocessed testing tweets
    @return
        dictionary of tweets
'''
def load_testing():
    tweets_test = []
    for i in range(3):
        with open("pickle/tweets_test{}_pp.p".format(i), "rb") as f:
            tweets_test.append( pickle.load(f) )
    return tweets_test


'''
    Returns sentiment keyword (positive, negative, neutral) from integer (1,-1,0)
    @param
        integer "code"
    @return
        sentiment
'''
def sentiment_from_code(code):
    if code == -1:
        return 'negative'
    elif code == 0:
        return 'neutral'
    else:
        return 'positive'
