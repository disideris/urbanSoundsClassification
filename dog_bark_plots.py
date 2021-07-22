from librosa import display
import librosa
import numpy as np
import matplotlib.pyplot as plt


# Feature extraction of dog bark
y,sr=librosa.load("/Users/dimitris/Desktop/urban_sound_dataset/UrbanSound8K/audio/fold5/100032-3-0-0.wav")
mfccs = librosa.feature.mfcc(y, sr, n_mfcc=40)
mel_spectrogram =librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40)
chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40)
chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40)
print(mel_spectrogram.shape, chroma_stft.shape, chroma_cq.shape, chroma_cens.shape, mfccs.shape)

#MFCC of dog bark
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('Dog Bark MFCC ')
plt.tight_layout()
plt.show()

#Melspectrogram of a dog bark
plt.figure(figsize=(10,4))
librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Dog Bark Mel Spectrogram')
plt.tight_layout()
plt.show()

#Chromagram of dog bark
plt.figure(figsize=(10,4))
librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Dog Bark  Chromagram')
plt.tight_layout()
plt.show()

#Chroma cqt of a dog bark
plt.figure(figsize=(10,4))
librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Dog Bark  Chroma CQT')
plt.tight_layout()
plt.show()

#Chroma cens of a dog bark
plt.figure(figsize=(10,4))
librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Dog Bark Chroma Cens')
plt.tight_layout()
plt.show()