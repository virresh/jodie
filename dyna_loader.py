import pandas as pd
import os
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import scale
import pickle

def load_network(time_scaling = True):
    location = os.environ["PROCESSED_INTERACTIONS"]
    itbloc = os.environ["ID_TO_BODY"]
    print("Loading network from {} and {}".format(location, itbloc))
    df = pd.read_pickle(location)
    with open(itbloc, "rb") as f:
        id_t_b = pickle.load(f)

    timestamps = []
    original_poster = []
    for i, interaction in df.iterrows():
        timestamps.append(id_t_b[interaction['comment_id']]['creation_date'])
        try:
            original_poster.append(id_t_b[interaction['submission_id']['post_id']]['owner']['user_id'])
        except:
            original_poster.append(None)
    
    #### Add timestamp and original poster information in the dataframe
    df['timestamp'] = timestamps
    df['original_poster'] = original_poster
    df.sort_values('timestamp')
    
    #### Seperate sequences in order of timestamp
    user_sequence = df['user_id'].values
    op_sequence = df['original_poster'].values
    item_sequence = df['item'].values
    timestamp_sequence = df['timestamp'].values
    submission_sequence = df['submission_id'].values
    feature_sequence = df.drop(
        ['user_id','item','comment_id','submission_id', 'timestamp', 'original_poster'],
        axis=1
    ).values
    y_true_labels = np.ones(len(user_sequence))
    
    #### Setup starting timestamp so that timestamps can be normalised (== current time - starting time)
    start_timestamp = np.min(timestamp_sequence)
    timestamp_sequence = timestamp_sequence - start_timestamp
    
    print("Formating item sequence")
    #### Assign every item a normalised item id
    #### and compute the time delta == time elapsed since last interaction with this item
    nodeid = 0
    item2id = {}
    item_timedifference_sequence = []
    item_current_timestamp = defaultdict(float)
    for cnt, item in enumerate(item_sequence):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        item_timedifference_sequence.append(timestamp - item_current_timestamp[item])
        item_current_timestamp[item] = timestamp
    num_items = len(item2id)
    item_sequence_id = [item2id[item] for item in item_sequence]

    
    print("Formating user sequence")
    #### Assign every user a normalised user id
    #### and compute the time delta == time elapsed since last interaction with this item
    nodeid = 0
    user2id = {}
    user_timedifference_sequence = []
    user_current_timestamp = defaultdict(float)
    user_previous_itemid_sequence = []
    user_latest_itemid = defaultdict(lambda: num_items)
    for cnt, user in enumerate(user_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        user_timedifference_sequence.append(timestamp - user_current_timestamp[user])
        user_current_timestamp[user] = timestamp
        user_previous_itemid_sequence.append(user_latest_itemid[user])
        user_latest_itemid[user] = item2id[item_sequence[cnt]]
    num_users = len(user2id)
    user_sequence_id = [user2id[user] for user in user_sequence]
    
    print("Formatting original poster sequence")
    #### Original Poster to Item ID mapping not required
    ### simply compute timediffs
    op_timedifference_sequence = []
    op_current_timestamp = defaultdict(float)
    for cnt, user in enumerate(op_sequence):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_sequence[cnt]
        op_timedifference_sequence.append(timestamp - op_current_timestamp[user])
        op_current_timestamp[user] = timestamp
    op_sequence_id = [user2id[user] for user in op_sequence]
    
    if time_scaling:
        print("Scaling Timestamps")
        user_timedifference_sequence = scale(np.array(user_timedifference_sequence) + 1)
        item_timedifference_sequence = scale(np.array(item_timedifference_sequence) + 1)
    print("*** Network loading completed ***\n\n")
    return [user2id, user_sequence_id, user_timedifference_sequence, user_previous_itemid_sequence, op_sequence_id, op_timedifference_sequence, item2id, item_sequence_id, item_timedifference_sequence, timestamp_sequence, feature_sequence, y_true_labels]


if __name__ == '__main__':
    ret = load_network()
