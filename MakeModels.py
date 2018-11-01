import os
import librosa
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_a_file(file_path):
  n_mfcc = 20
  y, sr = librosa.load(file_path, mono=True, sr=None)
  y = y[::3]
  mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=n_mfcc)
  # If maximum length exceeds mfcc lengths then pad the remaining ones
  if (11 > mfcc.shape[1]):
    pad_width = 11 - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
  # Else cutoff the remaining parts
  else:
    mfcc = mfcc[:, :11]
    
  return mfcc
def load(dir_path, label):
  mfcc_vectors = []
  genre_y = np.zeros((0, 1), dtype='int')
    
  files = os.listdir(dir_path)
  for i, file in enumerate(files):
    file_path = dir_path + file
    mfcc = load_a_file(file_path)
    mfcc_vectors.append(mfcc)
    genre_y = np.vstack((genre_y, label))
    print(f'{i+1}/{len(files)} loaded: {file_path}')
  
  genre_x = np.array(mfcc_vectors)
  return genre_x, genre_y

if __name__ == '__main__':
    # *---------------------------------ここを変えてください---------------------------------*
    # ファイルパスはこの.pyファイルからの相対パスで書いてください。
    # 引数二つ目のラベルは全角以外にしてください。
    # REEDME.mdを参考にしてください。
    # *------------------------------------------------------------------------------------*
    yes_x, yes_y = load('/wav/yes', '0')
    no_x, no_y = load('/wav/no', '1')

    X = np.r_[yes_x, no_x]
    Y = np.r_[yes_y, no_y]
    
    X_train = X.reshape(X.shape[0], 20, 11, 1)
    X_test = X.reshape(X.shape[0], 20, 11, 1)
    Y_train_hot = to_categorical(Y)
    Y_test_hot = to_categorical(Y)

    model = Sequential()
    model.add(Conv2D(32, (2, 2), input_shape=(20, 11, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (2, 2)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train_hot, batch_size=5, epochs=30, verbose=1, validation_data=(X_test, Y_test_hot))
    score = model.evaluate(X_test, Y_test_hot, batch_size=128)

    #モデル名も保存場所を変えても構いませんが、あとでmodel_loadする際変えてください。
    model.save('model_kutibue.h5')
    print(score[1])