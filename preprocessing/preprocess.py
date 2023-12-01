import pandas as pd
import numpy as np


def preprocess(test_part=0.2):
    data = pd.read_csv("data/raw/u.data", sep="\t", header=None)

    info = pd.read_csv("data/raw/u.info", sep=" ", header=None)
    num_users = info[0][0]
    num_items = info[0][1]

    # filter data with rating <= 3
    data = data[data[2] > 3]
    data = data.reset_index(drop=True)

    # 'shuffle' the data
    indexes = np.random.permutation(len(data))

    train_matrix = np.zeros((num_users, num_items), dtype=int)
    for i in indexes[:-round(len(data) * test_part)]:
        user_ind = data[0][i] - 1
        item_ind = data[1][i] - 1

        train_matrix[user_ind][item_ind] = 1

    test_matrix = np.zeros((num_users, num_items), dtype=int)
    for i in indexes[-round(len(data) * test_part):]:
        user_ind = data[0][i] - 1
        item_ind = data[1][i] - 1

        test_matrix[user_ind][item_ind] = 1
    
    np.save("data/interim/train.npy", train_matrix)
    np.save("data/interim/test.npy", test_matrix)


if __name__ == "__main__":
    np.random.seed(126)
    preprocess()
