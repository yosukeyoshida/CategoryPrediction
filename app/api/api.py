"""
API endpoint for CategoryPrediction
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../config')
import config

import json
import falcon
import numpy as np
import re

from category_prediction import CategoryPrediction

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class CategoryPredictionResource(object):
    def on_post(self, req, resp):
        body = req.stream.read()
        word_list = body.decode('utf-8')
        category_predict = CategoryPrediction(word_list)
        list = category_predict.predict_proba()
        msg = {
            "proba": list
        }
        resp.body = json.dumps(msg)

app = falcon.API()
app.add_route("/category_prediction/proba", CategoryPredictionResource())
