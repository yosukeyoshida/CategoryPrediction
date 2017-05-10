"""
Load training data from file
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../config')
import config
import json
import numpy as np

class DocumentReader:
    @classmethod
    def load_texts(cls):
      data = []
      label = []
      ids = []
      with open(config.TRAIN_DATA_PATH, 'r') as file:
          for line in file:
              id, category_id, text = line.rstrip().split("\t")
              words = json.loads(text)
              data.append(words)
              label.append(category_id)
              ids.append(id)
      data = np.array(data)
      label = np.array(label)
      ids = np.array(ids)
      return [data, label, ids]
