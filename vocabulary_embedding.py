import re
from sklearn.feature_extraction.text import HashingVectorizer

def lemmas_to_strings(x):
    x_list = [ " ".join(x.get(key)['tweet']) for key in x]
    y = [ x.get(key)['sentiment'] for key in x]

    pattern = re.compile(r"< ([a-z]+) >")
    x_list = [pattern.sub(r"<\1>", t) for t in x_list ]

    return x_list,y


def hashvectorizer(train_list, ngram_range=(1,2)):
    vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
                               ngram_range = (1,2))

    X_train = vectorizer.transform(train_list)
    return X_train, vectorizer

# def hashvectorizer(train_list, test_list, ngram_range=(1,2)):
#
#     vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False,
#                                ngram_range = (1,2))
#
#     X_train = vectorizer.transform(train_list)
#     X_test = []
#     for i in range(3):
#         X_test.append( vectorizer.transform(test_list[i]) )
#
#     return X_train, X_test
