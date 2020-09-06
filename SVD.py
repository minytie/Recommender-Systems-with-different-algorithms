from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
import json
from tqdm import tqdm

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')


for n_factors in tqdm([10 * i for i in range(1,20)],desc= "running SVD : "):

	# Use the famous SVD algorithm.
	algo = SVD(n_factors =  n_factors)

	# Run 5-fold cross-validation and print results.
	performance = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, n_jobs = -1 ,verbose=True)

	for key in performance:
		performance[key] = list(performance[key])

	with open("evaluations/SVD_Nfactors_%d.json" % n_factors,"w") as f:
		f.write(json.dumps(performance))
