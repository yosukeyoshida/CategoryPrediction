"""
Preprocessing corpus
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../config')
import config

from gensim import corpora, matutils, models

class Preprocessor:
    dictionary = corpora.Dictionary.load_from_text(config.DICTIONARY_PATH)

    def __init__(self, documents):
        bow_docs = []
        for document in documents:
            bow_doc = self.dictionary.doc2bow(document)
            bow_docs.append(bow_doc)
        self.lsi_model = models.LsiModel(bow_docs, num_topics=config.NUM_TOPICS, id2word=self.dictionary)
        print(self.lsi_model.print_topics(num_topics=20))

    def data2dense(self, data):
        ret = []
        for t in data:
            bow_doc = self.dictionary.doc2bow(t)
            lsi_doc = self.lsi_model[bow_doc]
            dense = list(matutils.corpus2dense([lsi_doc], num_terms=config.NUM_TOPICS).T[0])
            ret.append(dense)
        return ret
