import numpy as np
import math
import pandas as pd
    
def train_test_split(data, ratio=0.75, seed=42):
    train_list = []
    test_list = []

    grouped = data.groupby('user')
    users = list(grouped.groups.keys())

    np.random.seed(seed)
    np.random.shuffle(users)

    cold_start_number = int(len(users) * 0.2)
    for user in users[:cold_start_number]:
        group = grouped.get_group(user)
        if len(group) < 2:
            train_list.append(group)
        else:
            test_sample = group.sample(n=int(ratio * len(group)), random_state=seed)
            train_sample = group.drop(test_sample.index)
            train_list.append(train_sample)
            test_list.append(test_sample)

    for user in users[cold_start_number:]:
        group = grouped.get_group(user)
        if len(group) < 2:
            train_list.append(group)
        else:
            test_sample = group.sample(n=int((1 - ratio) * len(group)), random_state=seed)
            train_sample = group.drop(test_sample.index)
            train_list.append(train_sample)
            test_list.append(test_sample)

    train_data = pd.concat(train_list, ignore_index=True)
    test_data = pd.concat(test_list, ignore_index=True)
    return train_data, test_data
    
