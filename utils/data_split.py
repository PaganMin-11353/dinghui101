import pandas as pd
import numpy as np

def train_test_split(data, ratio=0.75, seed=42):
    train_list = []
    test_list = []

    # Group data by users
    grouped = data.groupby('user')
    users = list(grouped.groups.keys())

    # Shuffle users
    np.random.seed(seed)
    np.random.shuffle(users)

    # Divide users into cold-start and regular users
    cold_start_number = int(len(users) * 0.2)
    cold_start_users = users[:cold_start_number]
    regular_users = users[cold_start_number:]

    # Split cold-start users
    for user in cold_start_users:
        group = grouped.get_group(user)
        if len(group) < 2:
            train_list.append(group)  # If not enough data, add to train set
        else:
            test_sample = group.sample(n=int((1 - ratio) * len(group)), random_state=seed)
            train_sample = group.drop(test_sample.index)
            train_list.append(train_sample)
            test_list.append(test_sample)

    # Split regular users
    for user in regular_users:
        group = grouped.get_group(user)
        if len(group) < 2:
            train_list.append(group)  # If not enough data, add to train set
        else:
            test_sample = group.sample(n=int((1 - ratio) * len(group)), random_state=seed)
            train_sample = group.drop(test_sample.index)
            train_list.append(train_sample)
            test_list.append(test_sample)

    # Combine train and test lists
    train_data = pd.concat(train_list, ignore_index=True)
    test_data = pd.concat(test_list, ignore_index=True)

    # Ensure all items are in the training set
    all_items = set(data['item'])
    train_items = set(train_data['item'])

    # Find missing items in the training set
    missing_items = all_items - train_items

    # Add at least one interaction for each missing item to the training set
    for item in missing_items:
        item_data = test_data[test_data['item'] == item]
        if not item_data.empty:
            # Move one interaction back to the training set
            interaction = item_data.iloc[0]
            train_data = pd.concat([train_data, pd.DataFrame([interaction])], ignore_index=True)
            test_data = test_data.drop(interaction.name)

    return train_data, test_data