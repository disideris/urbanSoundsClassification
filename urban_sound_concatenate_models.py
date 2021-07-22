import librosa
import pandas as pd
from tqdm import tqdm
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from librosa import display
from pathlib import Path

# Extract mfcc audio features from an audio file
def extract_mfcc_audio_feature(audio_file):
    pad = 174
    y, sr = librosa.load(audio_file, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    new_pad = pad - mfccs.shape[1]
    mfccs =  np.pad(mfccs, pad_width=((0, 0), (0, new_pad)), mode='constant')
    return mfccs

# Extract mel audio features from an audio file
def extract_mel_audio_feature(audio_file):
    pad = 174
    y, sr = librosa.load(audio_file, res_type='kaiser_fast')
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000)
    new_pad = pad - mel_spectrogram.shape[1]
    mel_spectrogram = np.pad(mel_spectrogram, pad_width=((0, 0), (0, new_pad)), mode='constant')
    return mel_spectrogram

# Extract chroma cqt audio features from an audio file
def extract_cqt_audio_feature(audio_file):
    pad = 174
    y, sr = librosa.load(audio_file, res_type='kaiser_fast')
    chroma_cq =librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40)
    new_pad = pad - chroma_cq.shape[1]
    chroma_cq = np.pad(chroma_cq, pad_width=((0, 0), (0, new_pad)), mode='constant')
    return chroma_cq

# Extract chroma cens audio features from an audio file
def extract_cens_audio_feature(audio_file):
    pad = 174
    y, sr = librosa.load(audio_file, res_type='kaiser_fast')
    chroma_cens =librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40)
    new_pad = pad - chroma_cens.shape[1]
    chroma_cens = np.pad(chroma_cens, pad_width=((0, 0), (0, new_pad)), mode='constant')
    return chroma_cens


# Read metadata of urban sound dataset as pandas dataframe
metadata = pd.read_csv('/home/dimitris/Desktop/UrbanSound8K/metadata/UrbanSound8K'
                       '.csv')

print(metadata.head())
print(metadata.class_name.value_counts())

# Fill up feature_list List with features extraction of all audio files
path="/home/dimitris/Desktop/UrbanSound8K/audio/fold"

mfcc_feature_list = []
mel_feature_list = []
cqt_feature_list = []
cens_feature_list = []

for i in tqdm(range(len(metadata))):
    fold_no = str(metadata.iloc[i]["fold"])
    file = metadata.iloc[i]["slice_file_name"]
    class_label_id = metadata.iloc[i]["classID"]
    class_label = metadata.iloc[i]["class_name"]
    filename = path+fold_no+"/"+file
    mfcc_data = extract_mfcc_audio_feature(filename)
    mel_data = extract_mel_audio_feature(filename)
    cqt_data = extract_cqt_audio_feature(filename)
    cens_data = extract_cens_audio_feature(filename)
    mfcc_feature_list.append([mfcc_data, class_label])
    mel_feature_list.append([mel_data, class_label])
    cqt_feature_list.append([cqt_data, class_label])
    cens_feature_list.append([cens_data, class_label])


# Convert mfcc feature_list to pandas dataframe
features_dataframe = pd.DataFrame(mfcc_feature_list, columns=['feature', 'class_label'])

from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Convert features and class labels to numpy arrays
X = np.array(features_dataframe.feature.tolist())
y = np.array(features_dataframe.class_label.tolist())

# Encode classification labels
le = LabelEncoder()
categorical_y = to_categorical(le.fit_transform(y))

from sklearn.model_selection import train_test_split

# Split features array to train and test sets
x_train_mfcc, x_test_mfcc, y_train_mfcc, y_test_mfcc = train_test_split(X, categorical_y, test_size=0.2, random_state = 42)

# Convert mel feature_list to pandas dataframe
features_dataframe = pd.DataFrame(mel_feature_list, columns=['feature', 'class_label'])

# Convert features and class labels to numpy arrays
X = np.array(features_dataframe.feature.tolist())
y = np.array(features_dataframe.class_label.tolist())

# Encode classification labels
le = LabelEncoder()
categorical_y = to_categorical(le.fit_transform(y))

# Split features array to train and test sets
x_train_mel, x_test_mel, y_train_mel, y_test_mel = train_test_split(X, categorical_y, test_size=0.2, random_state = 42)

# Convert cqt feature_list to pandas dataframe
features_dataframe = pd.DataFrame(cqt_feature_list, columns=['feature', 'class_label'])

# Convert features and class labels to numpy arrays
X = np.array(features_dataframe.feature.tolist())
y = np.array(features_dataframe.class_label.tolist())

# Encode classification labels
le = LabelEncoder()
categorical_y = to_categorical(le.fit_transform(y))

# Split features array to train and test sets
x_train_cqt, x_test_cqt, y_train_cqt, y_test_cqt = train_test_split(X, categorical_y, test_size=0.2, random_state = 42)

# Convert cens feature_list to pandas dataframe
features_dataframe = pd.DataFrame(cqt_feature_list, columns=['feature', 'class_label'])

# Convert features and class labels to numpy arrays
X = np.array(features_dataframe.feature.tolist())
y = np.array(features_dataframe.class_label.tolist())

# Encode classification labels
le = LabelEncoder()
categorical_y = to_categorical(le.fit_transform(y))

# Split features array to train and test sets
x_train_cens, x_test_cens, y_train_cens, y_test_cens = train_test_split(X, categorical_y, test_size=0.2, random_state = 42)

from keras.models import Sequential, Model
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, merge
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

num_rows = 40
num_columns = 174
num_channels = 1

x_train_mfcc = x_train_mfcc.reshape(x_train_mfcc.shape[0], num_rows, num_columns, num_channels)
x_test_mfcc = x_test_mfcc.reshape(x_test_mfcc.shape[0], num_rows, num_columns, num_channels)

x_train_mel = x_train_mel.reshape(x_train_mel.shape[0], num_rows, num_columns, num_channels)
x_test_mel = x_test_mel.reshape(x_test_mel.shape[0], num_rows, num_columns, num_channels)

x_train_cqt = x_train_cqt.reshape(x_train_cqt.shape[0], num_rows, num_columns, num_channels)
x_test_cqt = x_test_cqt.reshape(x_test_cqt.shape[0], num_rows, num_columns, num_channels)

x_train_cens = x_train_cens.reshape(x_train_cens.shape[0], num_rows, num_columns, num_channels)
x_test_cens = x_test_cens.reshape(x_test_cens.shape[0], num_rows, num_columns, num_channels)

num_labels = categorical_y.shape[1]

# Mfcc CNN model architecture
mfcc_model = Sequential()
mfcc_model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
mfcc_model.add(MaxPooling2D(pool_size=2))
mfcc_model.add(Dropout(0.2))

mfcc_model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
mfcc_model.add(MaxPooling2D(pool_size=2))
mfcc_model.add(Dropout(0.2))

mfcc_model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
mfcc_model.add(MaxPooling2D(pool_size=2))
mfcc_model.add(Dropout(0.2))

mfcc_model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
mfcc_model.add(MaxPooling2D(pool_size=2))
mfcc_model.add(Dropout(0.2))
mfcc_model.add(GlobalAveragePooling2D())

mfcc_model.add(Dense(256, activation='relu'))

# Mel CNN model architecture
mel_model = Sequential()
mel_model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
mel_model.add(MaxPooling2D(pool_size=2))
mel_model.add(Dropout(0.2))

mel_model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
mel_model.add(MaxPooling2D(pool_size=2))
mel_model.add(Dropout(0.2))

mel_model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
mel_model.add(MaxPooling2D(pool_size=2))
mel_model.add(Dropout(0.2))

mel_model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
mel_model.add(MaxPooling2D(pool_size=2))
mel_model.add(Dropout(0.2))
mel_model.add(GlobalAveragePooling2D())

mel_model.add(Dense(256, activation='relu'))

# Cqt CNN model architecture
cqt_model = Sequential()
cqt_model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
cqt_model.add(MaxPooling2D(pool_size=2))
cqt_model.add(Dropout(0.2))

cqt_model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
cqt_model.add(MaxPooling2D(pool_size=2))
cqt_model.add(Dropout(0.2))

cqt_model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
cqt_model.add(MaxPooling2D(pool_size=2))
cqt_model.add(Dropout(0.2))

cqt_model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
cqt_model.add(MaxPooling2D(pool_size=2))
cqt_model.add(Dropout(0.2))
cqt_model.add(GlobalAveragePooling2D())

cqt_model.add(Dense(256, activation='relu'))

# Cens CNN model architecture
cens_model = Sequential()
cens_model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
cens_model.add(MaxPooling2D(pool_size=2))
cens_model.add(Dropout(0.2))

cens_model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
cens_model.add(MaxPooling2D(pool_size=2))
cens_model.add(Dropout(0.2))

cens_model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
cens_model.add(MaxPooling2D(pool_size=2))
cens_model.add(Dropout(0.2))

cens_model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
cens_model.add(MaxPooling2D(pool_size=2))
cens_model.add(Dropout(0.2))
cens_model.add(GlobalAveragePooling2D())

cens_model.add(Dense(256, activation='relu'))

# Concatenation of models
concat = concatenate([mfcc_model.output, mel_model.output, cqt_model.output, cens_model.output])
concat_out = Dense(num_labels, activation='softmax')(concat)

# Merge models
merge_model = Model([mfcc_model.input, mel_model.input, cqt_model.input, cens_model.input], concat_out)

# CNN model compilation
merge_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# CNN model summary
merge_model.summary()

# Train CNN model - y_train_mfcc has the labels of each soundclip and is the same as y_train_mel, y_train_cqt and y_train_cens.
merge_model.fit([x_train_mfcc, x_train_mel, x_train_cqt, x_train_cens], y_train_mfcc, batch_size=256, epochs=150, validation_data=([x_test_mfcc, x_test_mel, x_test_cqt, x_test_cens], y_test_mfcc), verbose=1)

# Evaluate CNN model
score = merge_model.evaluate([x_train_mfcc, x_train_mel, x_train_cqt, x_train_cens], y_train_mfcc, verbose=0)
print("Training Accuracy: ", score[1])
score = merge_model.evaluate([x_test_mfcc, x_test_mel, x_test_cqt, x_test_cens], y_test_mfcc, verbose=0)
print("Testing Accuracy: ", score[1])

merge_model.save('/home/dimitris/Desktop/UrbanSound8K/urban_sound_cnn_model3.h5')
