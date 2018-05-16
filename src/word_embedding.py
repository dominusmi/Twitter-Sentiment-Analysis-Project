import numpy as np
from tqdm import tqdm

model = None
model_d = 0

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    with open(gloveFile,'r') as f:
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.",len(model)," words loaded!")
        return model

def tweet_to_matrix(tweet):
    matrix = np.zeros((len(tweet), model_d), dtype='float')
    for index,token in enumerate(tweet):
        row = model.get(token)

        if row is not None:
            matrix[index,:] = row
        else:
            matrix[index,:] = np.zeros((1,model_d), dtype='float')
    return matrix

def embed_tweets(tweets):
    n_features = 5
    _embedding = np.zeros( (len(tweets), (n_features)*model_d), dtype='float')
    for n, tweet in tqdm(enumerate(tweets), total=len(tweets)):
        # Transform tweet to matrix
        m = tweet_to_matrix(tweets[n])

        # Sum over rows
        a = np.sum(m, axis=0)
        a /= np.max(abs(a))

        # Average over rows
        b = np.mean(m, axis=0)

        # Maximum of columns
        c = np.max(m ,axis=0)

        # Minimum of columns
        d = np.min(m, axis=0)

        # Median
        e = np.median(m, axis=0)

        # Add to final set
        _embedding[n,:] = np.hstack( (a,b,c,d,e) )

    return _embedding
