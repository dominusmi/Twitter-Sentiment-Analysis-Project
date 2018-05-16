# Twitter Sentiment Analysis Project
### CS918: Natural Language Processing, _University of Warwick_

As part of the above mentioned module, we had to develop three distinct sentiment analysis classifier capable of labellign tweets as either _positive, neutral_ or _negative_. The tweet dataset and general project was heavily inspired by the [semeval](http://alt.qcri.org/semeval2017/) competition.

The classifiers were tested against evaluated according to the macro-averaged F1-score, which meant that the inbalance in the tweets labels (negatives were nearly half of positives) was quite an important obstacle.

A strong focus was put on trying to extract features from the tweets, and word embeddings were only added later on, due mostly to the fact that recall tends to be much higher on hand-made rule-based features, although this also meant a lot more time spent.

More information can be find in the [final report](https://github.com/dominusmi/Twitter-Sentiment-Analysis-Project/blob/master/FINAL%20REPORT.pdf).

Enjoy!

_Edoardo Barp, May 2018_
