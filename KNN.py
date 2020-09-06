from surprise import KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate
import json
from tqdm import tqdm

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

for k in tqdm([5 * i for i in range(1,20)],desc= "running KNN : "):

	# Use the famous SVD algorithm.
	algo = KNNBasic(k =  5)
	#algo.test()

	# Run 5-fold cross-validation and print results.
	performance = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, n_jobs = -1 ,verbose=True)

	for key in performance:
		performance[key] = list(performance[key])

	with open("evaluations/KNN_%d.json" % k,"w") as f:
		f.write(json.dumps(performance))
