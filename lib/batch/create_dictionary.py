import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../config')
import config
from gensim import corpora
import re

STOP_WORDS = [
    '[ぁ-ん]'
]

joined_stop_words = '|'.join(STOP_WORDS)

regex = '^(%s)$' % joined_stop_words

match = re.compile(regex)

with open(config.KEYWORD_DATA_PATH, 'r') as file:
    words = []
    for line in file:
        line = line.rstrip()
        if match.search(line) is None:
            keywords = [line]
            words.append(keywords)

dictionary = corpora.Dictionary(words)
# dictionary.filter_extremes(no_below=20, no_above=0.5)
dictionary.save_as_text(config.DICTIONARY_PATH)
