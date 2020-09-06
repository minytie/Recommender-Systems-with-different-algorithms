from surprise import NormalPredictor
from surprise import Dataset
from surprise.model_selection import cross_validate,train_test_split
import json
import numpy as np

arr = [0.3341045379638672, 0.3650245666503906, 0.3610353469848633, 0.38197827339172363, 0.3590390682220459]
print(np.array(arr).mean())
print(np.array(arr).std())