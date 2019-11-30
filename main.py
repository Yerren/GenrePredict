import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def display_mfcc(song):
    y, _ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(song)
    plt.tight_layout()
    plt.show()

def extract_features_song(f):
    y, _ = librosa.load(f)

    mfcc = librosa.feature.mfcc(y)
    mfcc /= np.amax(np.absolute(mfcc))

    return np.ndarray.flatten(mfcc)[:25000]

def generate_features_and_labels():
    all_features = []
    all_labels = []

    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    for genre in genres:
        sound_files = glob.glob("genres/" +genre+"/*.au")
        print("Processing %d songs in %s genre..." % (len(sound_files), genre))
        for f in sound_files:
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = tf.keras.utils.to_categorical(label_row_ids, len(label_uniq_ids))
    return np.stack(all_features), onehot_labels


def plot_history(histories, key='acc'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key],
                   '--', label=name.title()+' Val')
    plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0, max(history.epoch)])
  plt.show()

# features, labels = generate_features_and_labels()
#
# np.save("datasets/DatasetSave.npy", features)
# np.save("datasets/LabelsSave.npy", labels)

features = np.load("datasets/DatasetSave.npy")
labels = np.load("datasets/LabelsSave.npy")

print(np.shape(features))
print(np.shape(labels))

training_split = 0.8

alldata = np.column_stack((features, labels))

np.random.shuffle(alldata)
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx, :], alldata[splitidx:, :]

print(np.shape(train))
print(np.shape(test))

train_input = train[:, :-10]
train_labels = train[:, -10:]

test_input = test[:, :-10]
test_labels = test[:, -10:]

print(np.shape(train_input))
print(np.shape(train_labels))

# activity_regularizer=keras.regularizers.l1(0.01)
model = keras.Sequential([
    keras.layers.Dense(128, input_dim=np.shape(train_input)[1], activation="relu", kernel_regularizer=keras.regularizers.l2(0.01), bias_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print(model.summary())

history = model.fit(train_input, train_labels, epochs=500, batch_size=64, validation_split=0.2)

loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

model.save_weights('./checkpoints/final_checkpoint')

print("finished")
print("loss: %.4f, accuracy: %.4f" % (loss, acc))

plot_history([(" ", history)])
