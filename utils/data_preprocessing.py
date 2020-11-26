import pandas as pd
import numpy as np
import glob
import pickle

def generate_vocab(training_path, min_freq=150, idx_start=0, idx_end = 65):
    
    print("GENERATING TRACK VOCAB DICTIONARY")
    train_input_logs = sorted(glob.glob(training_path + "log_0*.csv"))
    
    for i in range(idx_start, idx_end):
        
        print(train_input_logs[i])
        df = pd.read_csv(train_input_logs[i])
        track_counts = df.groupby("track_id_clean")['track_id_clean'].count()
        if i == 0:
            all_track_counts = track_counts.copy()
        else:
            all_track_counts = all_track_counts.append(track_counts)
            all_track_counts = all_track_counts.groupby('track_id_clean').sum()

    all_track_counts = all_track_counts[all_track_counts>=min_freq]
    
    track_vocab = {'pad':0, 'mask':1}
    track_ids = {track_id: i+len(track_vocab) for i, track_id in enumerate(all_track_counts.index)}
    track_vocab.update(track_ids)
    
    return track_vocab

def generate_training_data(training_path, track_vocab, train_input_logs, idx_start=0, idx_end = 65):

    all_session_tracks = []
    all_session_skips = []
        
    for i in range(idx_start, idx_end):
        print(train_input_logs[i])
        df = pd.read_csv(train_input_logs[0])
        df['skip_2'] = df['skip_2'].astype(int)

        #-1 for UNK songs not in filtered Track Vocabulary
        df['track_id_clean'] = df['track_id_clean'].map(track_vocab).fillna(-1).astype(int)
        unk_session_ids = set(df[df['track_id_clean']==-1]['session_id'].unique())
        full_session_ids = df.groupby('session_id')['track_id_clean'].count()
        full_session_ids = set(full_session_ids[full_session_ids==20].index)

        #filtering sessions with less than 20 length and which have any UNK song
        df = df[~df['session_id'].isin(unk_session_ids)]
        df = df[df['session_id'].isin(full_session_ids)]
        
        #saving to list format. return df if you want the raw sessions data instead.
        session_tracks = df.groupby("session_id")['track_id_clean'].apply(list)
        session_skips = df.groupby("session_id")['skip_2'].apply(list)
        session_skips = list(session_skips.values)
        session_tracks = list(session_tracks.values)

        all_session_tracks.extend(session_tracks)
        all_session_skips.extend(session_skips)
    
    return all_session_tracks, all_session_skips, df


def generate_track_features_embedding_weights(track_vocabs):
    #get embedding data
    #filter for only relevant embedding
    #create embedding matrix with row index corresponding to the track id value in the track vocab
    
    print("GENERATING TRACK FEATURE EMBEDDINGS")
    track1 = pd.read_csv("../raw_data/track_features/tf_000000000000.csv")
    track2 = pd.read_csv("../raw_data/track_features/tf_000000000001.csv")
    tracks = track1.append(track2, ignore_index=True)
    tracks_mapping = pd.DataFrame.from_dict(track_vocabs, orient='index', columns = ['track_index']).reset_index()
    tracks = tracks_mapping.merge(tracks, how='inner', left_on='index', right_on='track_id').sort_values('track_index')
    track_embedding = np.zeros((len(track_vocabs), 26))

    #first two rows of the embedding are for pad and mask tokens. only keeping numeric columns
    # note that i also L2 normalize this embedding prior to training in pytorch model itself
    track_embedding[2:,:] = tracks.iloc[0,np.r_[3:17,19:31]].values        
    
    return track_embedding


training_path = '../raw_data/training_set_0/'
testing_path = '../raw_data/training_set_9/'

#2 months (60 days) of train/test data and songs which were played globally at least 150 times in those 2 months
IDX_END = 60
MIN_FREQ = 150

print("------GENERATING TRAINING DATA------")

train_input_logs = sorted(glob.glob(training_path + "log_0*.csv"))
track_vocabs  = generate_vocab(training_path, min_freq=MIN_FREQ, idx_start=0, idx_end = IDX_END)

print("VOCAB SIZE")
print(len(track_vocabs))

f = open("../data/track_vocabs.pkl","wb")
pickle.dump(track_vocabs,f)
f.close()

track_embedding = generate_track_features_embedding_weights(track_vocabs)
np.save('../data/track_embedding.npy', track_embedding)

all_session_tracks, all_session_skips = generate_training_data(training_path, track_vocabs, train_input_logs, idx_start = 0, idx_end = IDX_END)

print("UNIQUE SESSIONS SIZE")
print(len(all_session_tracks))

f = open("../data/all_session_tracks_train.pkl","wb")
pickle.dump(all_session_tracks, f)
f.close()

f = open("../data/all_session_skips_train.pkl","wb")
pickle.dump(all_session_skips, f)
f.close()

print("------GENERATING TESTING DATA------")
test_input_logs = sorted(glob.glob(testing_path + "log_9*.csv"))
all_session_tracks, all_session_skips = generate_training_data(testing_path, track_vocabs, test_input_logs, idx_start = 0, idx_end = IDX_END)
f = open("../data/all_session_tracks_test.pkl","wb")
pickle.dump(all_session_tracks, f)
f.close()

print("UNIQUE SESSIONS SIZE")
print(len(all_session_tracks))

f = open("../data/all_session_skips_test.pkl","wb")
pickle.dump(all_session_skips, f)
f.close()
