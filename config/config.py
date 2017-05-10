import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/..'
DICTIONARY_PATH = '%s/data/dic.txt' % ROOT_DIR
MODELFILE_PATH = '%s/app/model/lr.pkl' % ROOT_DIR
KEYWORD_DATA_PATH = '%s/data/significant_terms.txt' % ROOT_DIR
TRAIN_DATA_PATH = '%s/data/article_train_data.tsv' % ROOT_DIR
LSI_PATH = '%s/app/model/lsi/lsi.pkl' % ROOT_DIR

MODEL_OPTIONS = {
    "C": 0.001,
    "penalty": 'l2'
}

NUM_TOPICS = 300

# multiple categories sample
CATEGORIES = {
    709: "Category_709",
    710: "Category_710",
    11:  "Category_11"
}

sys.path.append(ROOT_DIR + '/app/service')
