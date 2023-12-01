import pandas as pd
import numpy as np


def preprocess():
    make_train_data()
    make_benchmark_data()


def make_benchmark_data():
    test_data = pd.read_csv("data/raw/ua.test", sep="\t", header=None)

    info = pd.read_csv("data/raw/u.info", sep=" ", header=None)
    num_users = info[0][0]

    # text_matrix[user_ind] contains 10 test pairs (film and its rating)
    test_matrix = np.zeros((num_users, 10, 2))
    for i in range(len(test_data)):
        user_ind = test_data[0][i] - 1
        pair_ind = i % 10
        item_ind = test_data[1][i] - 1
        rating = test_data[2][i]

        test_matrix[user_ind][pair_ind][0] = item_ind
        test_matrix[user_ind][pair_ind][1] = rating

    np.save("benchmark/data/test.npy", test_matrix)


def make_train_data():
    train_data = pd.read_csv("data/raw/ua.base", sep="\t", header=None)

    info = pd.read_csv("data/raw/u.info", sep=" ", header=None)
    num_users = info[0][0]
    num_items = info[0][1]

    matrix = np.zeros((num_users, num_items))
    for i in range(len(train_data)):
        user_ind = train_data[0][i] - 1
        item_ind = train_data[1][i] - 1
        rating = train_data[2][i]

        matrix[user_ind][item_ind] = rating
    
    np.save("data/interim/train.npy", matrix)


if __name__ == "__main__":
    preprocess()
