import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../config')
import config

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

from document_reader import DocumentReader
from preprocessor import Preprocessor

if __name__ == "__main__":
    data, label, ids = DocumentReader.load_texts()
    preprocessor = Preprocessor(data)
    joblib.dump(preprocessor.lsi_model, config.LSI_PATH)
    data_lsi = preprocessor.data2dense(data)
    
    penalty = config.MODEL_OPTIONS["penalty"]
    C = config.MODEL_OPTIONS["C"]
    print("training... C=%s penalty=%s" % (C, penalty))
    estimator = LogisticRegression(penalty=penalty, C=C)
    estimator.fit(data_lsi, label)
    joblib.dump(estimator, config.MODELFILE_PATH)
