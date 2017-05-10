"""
Category Prediction
Input:   Words
Output:  Category probability descending
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../config')
import config

from gensim import corpora, matutils
from sklearn.externals import joblib
import json
import sys

class CategoryPrediction:
    dictionary = corpora.Dictionary.load_from_text(config.DICTIONARY_PATH)
    estimator = joblib.load(config.MODELFILE_PATH)
    lsi_model = joblib.load(config.LSI_PATH)

    def __init__(self, word_list):
        self.word_list = word_list

    def create_data(self):
        words = json.loads(self.word_list)
        bow_doc = self.dictionary.doc2bow(words)
        lsi_doc = self.lsi_model[bow_doc]
        dense = list(matutils.corpus2dense([lsi_doc], num_terms=config.NUM_TOPICS).T[0])
        return dense

    def predict_proba(self):
        data = self.create_data()
        predict_proba = self.estimator.predict_proba(data)[0]
        dic = {}
        for category_id, proba in zip(self.estimator.classes_, predict_proba):
          dic[category_id] = proba

        sorted_array = sorted(dic.items(), key=lambda x:x[1], reverse=True)

        ret = []
        for t in sorted_array:
          ret.append([t[0], t[1]])
        return ret

if __name__ == "__main__":
    param = sys.argv
    word_list = param[1]
    category_predict = CategoryPrediction(word_list)
    print(category_predict.predict_proba())
