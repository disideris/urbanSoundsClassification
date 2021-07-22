import librosa
import pandas as pd
from tqdm import tqdm
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from librosa import display
from pathlib import Path

# Extract selected audio features from an audio file
def extract_audio_feature(audio_file):
    pad = 174
    y, sr = librosa.load(audio_file, res_type='kaiser_fast')
    # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000)
    chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40)
    new_pad = pad - chroma_cq.shape[1]
    return np.pad(chroma_cq, pad_width=((0, 0), (0, new_pad)), mode='constant')

# Predict and print class of an unlabeled audio file
def predict_class_of_audio_file(audio_file):
    pred = extract_audio_feature(audio_file)
    pred = pred.reshape(1, num_rows, num_columns, num_channels)
    pred_class = model.predict_classes(pred)
    pred_class_trans = le.inverse_transform(pred_class)
    print("The predicted class is:", pred_class_trans[0])

# Read metadata of urban sound dataset as pandas dataframe
metadata = pd.read_csv('/home/dimitris/Desktop/UrbanSound8K/metadata/UrbanSound8K.csv')

print(metadata.head())
print(metadata.class_name.value_counts())

# Fill up feature_list List with features extraction of all audio files
path="/home/dimitris/Desktop/UrbanSound8K/audio/fold"
feature_list = []
for i in tqdm(range(len(metadata))):
    fold_no = str(metadata.iloc[i]["fold"])
    file = metadata.iloc[i]["slice_file_name"]
    class_label_id = metadata.iloc[i]["classID"]
    class_label = metadata.iloc[i]["class_name"]
    filename = path+fold_no+"/"+file
    data = extract_audio_feature(filename)
    feature_list.append([data, class_label])


# Convert feature_list to pandas dataframe
features_dataframe = pd.DataFrame(feature_list, columns=['feature', 'class_label'])

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
x_train, x_test, y_train, y_test = train_test_split(X, categorical_y, test_size=0.2, random_state = 42)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
#
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

num_rows = 40
num_columns = 174
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = categorical_y.shape[1]
filter_size = 2

# CNN model architecture
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))

# CNN model compilation
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# CNN model summary
model.summary()

# Train CNN model
model.fit(x_train, y_train, batch_size=256, epochs=150, validation_data=(x_test, y_test), verbose=1)

# Evaluate CNN model
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])
score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])

model.save('/home/dimitris/Desktop/UrbanSound8K/urban_sound_cnn_model1.h5')

# Predict unlabeled audio files
from pathlib import Path
import os

path_str="/home/dimitris/Desktop/urban_sound_dataset/test2"
pathlist = Path(path_str).glob('**/*.wav')
for path in pathlist:
     audio_file_path = str(path)
     head, tail = os.path.split(audio_file_path)
     print(tail)
     predict_class_of_audio_file(audio_file_path)
     print("\n")
