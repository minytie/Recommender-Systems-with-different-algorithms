from surprise import NormalPredictor
from surprise import Dataset
from surprise.model_selection import cross_validate
import json

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# Use the famous SVD algorithm.
algo = NormalPredictor()

# Run 5-fold cross-validation and print results.
performance = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, n_jobs = -1 ,verbose=True)

for key in performance:
	performance[key] = list(performance[key])


with open("evaluations/random.json","w") as f:
	f.write(json.dumps(performance))
