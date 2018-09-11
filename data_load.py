# -*- coding: utf-8 -*-
"""
Spyder Editor
Lect 1 : Peak finding
"""

from lda_util import *
from textblob import TextBlob 
import numpy as np

in_dir = '/Users/leihao/Downloads/Data/'
file = in_dir + 'nasdaq.db'

start_date, end_date = '2016-01-01', '2016-01-02'
news_articles = article_extractor(file, start_date, end_date)

articles = news_articles.article

#articles[1].split(sep='The next billion-dollar iSecre')[0]
art_senti = []
for art in articles:
    art = art.split(sep='The next billion-dollar iSecre')[0]
    art_senti.append(TextBlob(art).sentiment)

sent_senti=[]    
sentences = TextBlob(articles[1]).sentences
for sent in sentences:
    sent_senti.append(sent.sentiment.polarity)

art_senti = [ item if item >=1.0 for item in sent_senti ]

