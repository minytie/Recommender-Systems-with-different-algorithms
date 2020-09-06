import torch
from torch.nn import Module
from surprise import Trainset, Dataset
from surprise.model_selection import  KFold, train_test_split
from typing import List, Tuple, Generator
from random import shuffle
from tqdm import  tqdm
import time
import torch.optim as optim
import numpy as np
import json

input_tensor = torch.tensor

class DeepNeturalNetwork(Module):

	def __init__(self,embedding_dim : int,train_set : Trainset):
		super().__init__()
		self.train_set = train_set
		self.u = torch.nn.Embedding(num_embeddings = self.train_set.n_users + 1,
									embedding_dim = embedding_dim)
		self.m = torch.nn.Embedding(num_embeddings = self.train_set.n_items + 1,
									embedding_dim = embedding_dim)
		self.hidden = torch.nn.Sequential(
			torch.nn.Dropout(p = 0.02),
			torch.nn.Linear(embedding_dim * 2,out_features = 100,bias=True),
			torch.nn.ReLU(),
			torch.nn.Dropout(p = 0.25),
			torch.nn.Linear(in_features = 100, out_features=200,bias=True),
			torch.nn.ReLU(),
			torch.nn.Dropout(p = 0.5),
			torch.nn.Linear(in_features = 200, out_features=300,bias=True),
			torch.nn.ReLU()
		)
		self.fc = torch.nn.Linear(in_features=300,out_features=1,bias = True)

	def forward(self,users : torch.tensor , movies : torch.tensor):
		users_embedding  = self.u(users)
		movies_embedding = self.m(movies)
		input_features = torch.cat([users_embedding,movies_embedding],dim = -1)
		hidden = self.hidden(input_features)
		pred   = self.fc(hidden)
		return pred

	def loss_func(self,pred : torch.tensor,real : torch.tensor):

		loss = torch.nn.MSELoss(reduction="sum")
		return loss(pred,real)

	def predict(self,raw_uid,raw_iid):...



def trainsetBatchIter(trainset : Trainset,batch_size = 1000) -> Generator:

	p = 0
	ratings = list(trainset.all_ratings())
	while True:
		if p >= len(ratings):
			shuffle(ratings)
			p = 0

		batch_ratings = ratings[p : p + batch_size]
		uids = [] #user ids
		iids = [] #item ids
		rs   = [] #rating score
		for rating in batch_ratings:
			u,i,r = rating
			uids.append(u)
			iids.append(i)
			rs.append(r)
		yield (torch.tensor(uids,dtype= torch.long),
			   torch.tensor(iids, dtype=torch.long),
			   torch.tensor(rs, dtype=torch.float))
		p += batch_size

def testsetBatchIter(trainset : Trainset,testset : list,batch_size) -> Generator :
	p = 0
	while p < len(testset):
		batch_ratings = testset[p : p + batch_size]
		if len(batch_ratings) == 0:break
		uids = [] #user ids
		iids = [] #item ids
		rs   = [] #rating score
		known_users = []
		known_items = []
		for rating in batch_ratings:
			raw_u,raw_i,r = rating
			try:
				uid = trainset.to_inner_uid(raw_u)
				know_user = 1
			except:
				uid = trainset.n_users
				know_user = 0
			try:
				iid = trainset.to_inner_iid(raw_i)
				know_item = 1
			except:
				iid = trainset.n_items
				know_item = 0
			uids.append(uid)
			iids.append(iid)
			rs.append(r)
			known_users.append(know_user)
			known_items.append(know_item)

		yield (torch.tensor(uids,dtype= torch.long),
			   torch.tensor(iids, dtype=torch.long),
			   torch.tensor(rs, dtype=torch.float),
			   torch.tensor(known_users, dtype=torch.long),
			   torch.tensor(known_items, dtype=torch.long))
		p += batch_size

if __name__ == "__main__":
	data = Dataset.load_builtin('ml-100k')
	kf = KFold(n_splits=5)
	for embedding_dim in [75,100,150,200,250]:
		lr = 2e-4
		wd = 1e-5
		batch_size = 200
		n_epoches = 15
		now_fold = 0
		history  = {}
		for trainset, testset in kf.split(data):
			net = DeepNeturalNetwork(embedding_dim=embedding_dim, train_set=trainset)
			# training parameters
			# use GPU if available
			identifier = 'cuda:0' if torch.cuda.is_available() else 'cpu'
			device = torch.device(identifier)
			optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
			train_batch_num = int(len(list(trainset.all_ratings())) / batch_size)
			lr_history = {"test_rmse" : [],"test_mae":[],"test_time" : []}
			net.to(device)
			for epoch in tqdm(list(range(n_epoches)),desc = "training net work of fold : %d " % now_fold):
				#train
				losses = []
				train_batch_iter = trainsetBatchIter(trainset,batch_size = 10)
				for idx in range(train_batch_num):
					batch = train_batch_iter.__next__()
					optimizer.zero_grad()
					uids,iids,rs = batch
					uids = uids.to(device)
					iids = iids.to(device)
					rs = rs.to(device)
					estimate_rs = net(uids,iids)
					estimate_rs = estimate_rs.view(len(estimate_rs))
					loss = net.loss_func(estimate_rs,rs)
					loss.backward()
					optimizer.step()
					losses.append(loss.cpu().item())
					print("\rfold %d epoch %d (process : %f): loss :%f " % (now_fold,epoch,idx / train_batch_num ,loss.cpu().item()),end = "",flush=True)

				print("\rfold %d epoch %d : loss :%f " % (now_fold,epoch,sum(losses)/len(losses)))
				#eval
				rmse = 0
				mae  = 0
				start_time = time.time()
				count = 0
				with torch.no_grad():
					for batch in testsetBatchIter(trainset,testset,batch_size):
						uids,iids,rs,known_users,known_items = batch
						uids = uids.to(device)
						iids = iids.to(device)
						rs   = rs.to(device)
						estimate_rs = net(uids,iids)
						estimate_rs = estimate_rs.view(len(estimate_rs))
						errors = (estimate_rs - rs)
						errors[(known_users == 0) | (known_items == 0)] = 0
						errors = errors.cpu().numpy()
						for i in range(len(errors)):
							error = errors[i]
							mae += abs(error)
							rmse += error * error
							count += 1

				rmse = np.sqrt(rmse/count)
				mae  = mae / count
				print("\rfold %d epoch %d : rmse :%f | mae:%f " % (now_fold,epoch,rmse,mae))
				time_usage = time.time() - start_time
				lr_history["test_rmse"].append(rmse)
				lr_history["test_mae"].append(mae)
				lr_history["test_time"].append(time_usage)
			torch.cuda.empty_cache()
			history[now_fold] = lr_history
			now_fold += 1

		performance = {"test_rmse": [], "test_mae": [], "test_time": []}
		for key in history:
			lr_history = history[key]
			performance["test_rmse"].append(min(lr_history["test_rmse"]))
			performance["test_mae"].append(min(lr_history["test_mae"]))
			performance["test_time"].append(min(lr_history["test_time"]))

		with open("evaluations/dnn_embedding_dim_%d_lr_history.json" % embedding_dim,"w") as f:
			f.write(json.dumps(history))
		with open("evaluations/dnn_embedding_dim_%d.json" % embedding_dim,"w") as f:
			f.write(json.dumps(performance))