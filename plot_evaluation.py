import matplotlib.pyplot as plt
import json
def plot_model_selection(alg : str):
	assert alg in ["knn","svd","matrix_factorization","dnn"]
	plt.figure()
	if alg == "knn":
		ks = [5 * i for i in range(1,20)]
		rmses = []
		for k in ks:
			with open("evaluations/KNN_%d.json" % k) as f:
				performance = json.loads(f.read())
				rmses.append(sum(performance["test_rmse"])/len(performance["test_rmse"]))
		plt.plot(ks,rmses,label = "test rmse")
		plt.xlabel("k")
		plt.title("test rmse of different k")
	if alg == "svd":
		num_factors = [10 * i for i in range(1,20)]
		rmses = []
		for num_factor in num_factors:
			with open("evaluations/SVD_Nfactors_%d.json" % num_factor) as f:
				performance = json.loads(f.read())
				rmses.append(sum(performance["test_rmse"])/len(performance["test_rmse"]))
		plt.plot(num_factors,rmses,label = "test rmse")
		plt.xlabel("num_factors")
		plt.title("test rmse of different num_factors")

	if alg == "matrix_factorization":
		betas = ["0.000000" , "0.000020","0.000040","0.000060","0.000080","0.000100","0.000120","0.000200","0.000500"]
		rmses = []
		for beta in betas:
			with open("evaluations/Matrix_factorization_beta_%s.json" % str(beta)) as f:
				performance = json.loads(f.read())
				rmses.append(sum(performance["test_rmse"])/len(performance["test_rmse"]))
		plt.plot(list(map(float,betas)),rmses,label = "test rmse")
		plt.xlabel("beta")
		plt.title("test rmse of different beta")
	if alg == "dnn":
		embedding_dims = [75, 100, 150, 200, 250]
		rmses = []
		for embedding_dim in embedding_dims:
			with open("evaluations/dnn_embedding_dim_%d.json" % embedding_dim) as f:
				performance = json.loads(f.read())
				rmses.append(sum(performance["test_rmse"])/len(performance["test_rmse"]))
		plt.plot(embedding_dims,rmses,label = "test rmse")
		plt.xlabel("beta")
		plt.title("test rmse of different embedding_dim")
	plt.legend(loc = "upper right")
	plt.show()
if __name__ == "__main__":
	for alg in ["knn","svd","matrix_factorization","dnn"]:
		plot_model_selection(alg)