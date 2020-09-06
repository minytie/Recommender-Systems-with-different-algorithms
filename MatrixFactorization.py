from surprise import AlgoBase
from surprise import Dataset
from surprise import Trainset
from surprise import PredictionImpossible
from surprise.model_selection import cross_validate, train_test_split, KFold
import numpy as np
import time
import json
from tqdm import tqdm

class Matrix_factorization:

    def __init__(self,num_factors : int):
        #init matrix
        self.num_factors = num_factors

    def fit(self,trainset : Trainset):

        self.trainset = trainset
        self.UserFeature = np.random.randn(trainset.n_users,self.num_factors)
        self.ItemFeature = np.random.randn(trainset.n_items,self.num_factors)
        self.rating_matrix = np.zeros((trainset.n_users,trainset.n_items))
        for rating in self.trainset.all_ratings():
            i, j, r = rating
            self.rating_matrix[i][j] = r
        print("loading dataset of %d users and %d items" % (trainset.n_users,trainset.n_items))

    def predict(self,uid,iid):
        try:
            uid = self.trainset.to_inner_uid(uid)
            iid = self.trainset.to_inner_iid(iid)
            return self.estimate_rating_matrix[uid][iid]
        except:
            return  -1

    def eval(self,testset : list):
        start_time = time.time()
        rmse = 0
        mae = 0
        count = 0
        for rating in testset:
            raw_uid,raw_iid,r = rating
            try:
                uid = self.trainset.to_inner_uid(raw_uid)
                iid = self.trainset.to_inner_iid(raw_iid)
            except:
                continue
            estimate_r = self.estimate_rating_matrix[uid][iid]
            error = estimate_r - r
            mae += abs(error)
            rmse += error * error
            count += 1

        rmse = np.sqrt(rmse / len(testset))
        mae  = mae / len(testset)
        time_usage = time.time() - start_time

        return rmse,mae,time_usage


    def train(self,steps=5000, alpha=0.0002,tol = 1e-3, beta=0.02,verbose = False):
        '''
        :param steps:
        :param alpha:
        :param beta:
        :return:
        '''
        self.ItemFeature = self.ItemFeature.T
        last_e = 100000
        for step in range(steps):
            #update matrix
            for rating in self.trainset.all_ratings():
                i,j,r = rating
                eij = r - np.dot(self.UserFeature[i,:],self.ItemFeature[:,j])
                self.UserFeature[i,:]  += 2 * eij * self.ItemFeature[:,j] * alpha - beta * self.UserFeature[i,:]
                self.ItemFeature[:,j]  += 2 * eij * self.UserFeature[i,:] * alpha - beta * self.ItemFeature[:,j]

            estimate_rating_matrix = self.UserFeature @ self.ItemFeature
            error_matrix = estimate_rating_matrix - self.rating_matrix
            error_matrix[self.rating_matrix == 0] = 0
            rmse_error = np.linalg.norm(error_matrix,ord = "fro") / np.sqrt(self.trainset.n_ratings)
            regu_error = np.linalg.norm(self.UserFeature,ord = "fro") ** 2 + np.linalg.norm(self.ItemFeature,ord = "fro")**2
            regu_error = regu_error * (beta/2)

            e =regu_error + rmse_error

            if verbose:
                print("\r step %d : Rmse error : %f | Regularization error : %f | Total error : %f" % (step,rmse_error,regu_error,e),end = "",flush=True)

            
            if last_e - e < tol:
                break
            last_e = e

        self.estimate_rating_matrix = self.UserFeature @ self.ItemFeature
        self.estimate_rating_matrix[self.estimate_rating_matrix >= 5] = 5
        self.estimate_rating_matrix[self.estimate_rating_matrix <= 0] = 0
        self.ItemFeature = self.ItemFeature.T

if __name__ == "__main__":
    # Load the movielens-100k dataset (download it if needed).
    data = Dataset.load_builtin('ml-100k')
    kf = KFold(n_splits=5)
    betas = [0.0002,0.0005,0.001]
    for beta in  tqdm(betas,desc = "running matrix factorization"):
        kf_results = []
        for trainset, testset in kf.split(data):
            alg = Matrix_factorization(num_factors=40)
            alg.fit(trainset)
            alg.train(verbose=True, beta = beta, steps=100)
            res = alg.eval(testset)
            kf_results.append(res)
            print(res)
        performance = {"test_rmse":[],"test_mae":[],"test_time":[]}
        for res in kf_results:
            performance["test_rmse"].append(res[0])
            performance["test_mae"].append(res[1])
            performance["test_time"].append(res[2])
        with open("evaluations/Matrix_factorization_beta_%f.json" % beta,"w") as f:
            f.write(json.dumps(performance))