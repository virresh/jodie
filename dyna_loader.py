import pandas as pd
import os
import numpy as np
import pickle

# def load_network():
df = pd.read_pickle(os.environ["PROCESSED_INTERACTIONS"])
with open(os.environ["ID_TO_BODY"], "rb") as f:
    id_t_b = pickle.load(f)

timestamps = []
for i, interaction in df.iterrows():
    timestamps.append(id_t_b[interaction['comment_id']]['creation_date'])
ordering = np.argsort(timestamps)
df['timestamp'] = ordering
df.set_index(ordering)
df.sort_index()
    # return df
