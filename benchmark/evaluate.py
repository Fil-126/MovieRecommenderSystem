import os
import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, AUC_at_k
from scipy.sparse import csr_matrix
import pandas as pd
import threadpoolctl


def evaluate(model):
    train_data = np.load("data/interim/train.npy")
    train_sparse = csr_matrix(train_data)
    
    test_data = np.load("data/interim/test.npy")
    test_sparse = csr_matrix(test_data)

    AUC_score = AUC_at_k(model, train_sparse, test_sparse, 10)
    precision = precision_at_k(model, train_sparse, test_sparse, 10)
    
    return AUC_score, precision
    


if __name__ == "__main__":
    threadpoolctl.threadpool_limits(1, "blas")

    eval_data = {
        "model_name": [],
        "AUC_score@10": [],
        "precision@10": [],
    }

    for file in os.listdir("models"):
        if file.endswith(".npz"):
            model = AlternatingLeastSquares.load("models/" + file)
            AUC_score, precision = evaluate(model)
            
            eval_data["model_name"].append(file[:-4])
            eval_data["AUC_score@10"].append(AUC_score)
            eval_data["precision@10"].append(precision)
    
    df = pd.DataFrame(eval_data)
    df.to_csv("benchmark/data/eval_results.csv", index=None)
