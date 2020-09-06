from surprise import AlgoBase, KNNBasic, SVD, NormalPredictor
from surprise import Dataset
from surprise import Trainset
from surprise import PredictionImpossible
from surprise.model_selection import cross_validate, train_test_split, KFold
import numpy as np
import time
import json
from MatrixFactorization import *
from DeepNN import *
from tqdm import tqdm
import pandas as pd
import os


data = Dataset.load_builtin('ml-100k')
trainset,testset = train_test_split(data)

normal = NormalPredictor()
normal.fit(trainset)
normal_prediction_seq = []
for r_u,r_i,r in tqdm(testset,desc="get prediction sequence of random algorithm"):
	res = estimate_rs = normal.predict(r_u,r_i)
	normal_prediction_seq.append(res.est)

if not os.path.exists("prediction_seqs/knn.json"):

	knn = KNNBasic(k = 90)
	knn.fit(trainset)
	KNN_prediction_seq = []
	for r_u,r_i,r in tqdm(testset,desc="get prediction sequence of knn"):
		res = estimate_rs = knn.predict(r_u,r_i)
		KNN_prediction_seq.append(res.est)
	with open("prediction_seqs/knn.json","w") as f:
		f.write(json.dumps(KNN_prediction_seq))
else:
	KNN_prediction_seq = json.loads(open("prediction_seqs/knn.json").read())

if not os.path.exists("prediction_seqs/svd.json"):

	svd = SVD(n_factors =  40)
	svd.fit(trainset)
	svd_prediction_seq = []
	for r_u,r_i,r in tqdm(testset,desc="get prediction sequence of svd"):
		res = estimate_rs = svd.predict(r_u,r_i)
		svd_prediction_seq.append(res.est)
	with open("prediction_seqs/svd.json","w") as f:
		f.write(json.dumps(svd_prediction_seq))
else:
	svd_prediction_seq = json.loads(open("prediction_seqs/svd.json").read())

if not os.path.exists("prediction_seqs/MF.json"):

	MF = Matrix_factorization(num_factors=40)
	MF.fit(trainset)
	MF.train(verbose=True, beta=0.0002, steps=100)
	MF_prediction_seq = []
	for r_u,r_i,r in tqdm(testset,desc="get prediction sequence of MF"):
		res = estimate_rs = MF.predict(r_u,r_i)
		MF_prediction_seq.append(res)
	with open("prediction_seqs/MF.json","w") as f:
		f.write(json.dumps(MF_prediction_seq))
else:
	MF_prediction_seq = json.loads(open("prediction_seqs/MF.json").read())

if not os.path.exists("prediction_seqs/NN.json"):
	NN_prediction_seq = []
	lr = 2e-4
	wd = 1e-5
	batch_size = 200
	n_epoches = 10
	net = DeepNeturalNetwork(embedding_dim=100, train_set=trainset)
	# training parameters
	# use GPU if available
	identifier = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	device = torch.device(identifier)
	optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
	train_batch_num = int(len(list(trainset.all_ratings())) / batch_size)
	net.to(device)
	for epoch in tqdm(list(range(n_epoches)), desc="training net work of fold : " ):
		# train
		losses = []
		train_batch_iter = trainsetBatchIter(trainset, batch_size=10)
		for idx in range(train_batch_num):
			batch = train_batch_iter.__next__()
			optimizer.zero_grad()
			uids, iids, rs = batch
			uids = uids.to(device)
			iids = iids.to(device)
			rs = rs.to(device)
			estimate_rs = net(uids, iids)
			estimate_rs = estimate_rs.view(len(estimate_rs))
			loss = net.loss_func(estimate_rs, rs)
			loss.backward()
			optimizer.step()
			losses.append(loss.cpu().item())
			print("\repoch %d (process : %f): loss :%f " % (epoch, idx / train_batch_num, loss.cpu().item()), end="", flush=True)

		print("\repoch %d : loss :%f " % (epoch, sum(losses) / len(losses)))
		# eval
	with torch.no_grad():
		for batch in testsetBatchIter(trainset, testset,batch_size):
			uids, iids, rs, known_users, known_items = batch
			uids = uids.to(device)
			iids = iids.to(device)
			rs = rs.to(device)
			estimate_rs = net(uids, iids)
			estimate_rs = estimate_rs.view(len(estimate_rs))
			estimate_rs[(known_users == 0) | (known_items == 0)] = -1
			estimate_rs = estimate_rs.cpu().numpy()
			for i in range(len(estimate_rs)):
				est_rs = estimate_rs[i]
				NN_prediction_seq.append(float(est_rs))
	with open("prediction_seqs/NN.json","w") as f:
		f.write(json.dumps(NN_prediction_seq))
else:
	NN_prediction_seq = json.loads(open("prediction_seqs/NN.json").read())

assert len(normal_prediction_seq) == len(KNN_prediction_seq)
assert len(KNN_prediction_seq) == len(svd_prediction_seq)
assert len(svd_prediction_seq) == len(MF_prediction_seq)
assert len(MF_prediction_seq) == len(NN_prediction_seq)

def cal_rmse(seq1 : list,seq2 : list) -> float:
	rmse = 0
	count = 0
	for i in range(len(seq1)):
		if seq1[i] == -1 or seq2[i] == -1 :continue
		dist = seq1[i] - seq2[i]
		rmse += dist * dist
		count += 1
	return np.sqrt(rmse / count)

algorithm_seqs = {
	"normal" : normal_prediction_seq,
	"knn" : KNN_prediction_seq,
	"svd" : svd_prediction_seq,
	"matrix_factorization" : MF_prediction_seq,
	"deep netural network" : NN_prediction_seq
}
algorithms = list(algorithm_seqs.keys())
comparision_table = [[0 for j in range(len(algorithms))] for i in range(len(algorithms))]
for i,alg_i in enumerate(algorithms):
	for j,alg_j in enumerate(algorithms):
		comparision_table[i][j] = cal_rmse(seq1 = algorithm_seqs[alg_i],
											seq2 = algorithm_seqs[alg_j])

df = pd.DataFrame(np.array(comparision_table))
df.columns = algorithms
df.index = algorithms
df["mean"] = df.mean(axis = 0)
df.to_csv("comparision_table.csv")
print(df)