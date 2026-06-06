import os
import sys
import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, MaxPooling3D,
    Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

BASE_PATH = "/kaggle/input/hsi-classification"
DATA_PATH = os.path.join(BASE_PATH, "train")
TEST_DATA_PATH = os.path.join(BASE_PATH, "test")
LABEL_FILE = os.path.join(BASE_PATH, "labels.csv")

print(f"Citire etichete din: {LABEL_FILE}")
try:
    labels_df = pd.read_csv(LABEL_FILE)
    labels_df = labels_df.set_index("filename")
except FileNotFoundError:
    print("EROARE: labels.csv nu a fost gasit!")
    sys.exit(1)

X_list = []
Y_list = []
num_fisiere_sarite = 0

for file_name in os.listdir(DATA_PATH):
    if file_name.endswith(".npy"):
        try:
            patch = np.load(os.path.join(DATA_PATH, file_name), allow_pickle=False)
            label = labels_df.loc[file_name, "label"]
            if label in [4, 5]:
                continue
            X_list.append(patch)
            Y_list.append(label)
        except Exception:
            num_fisiere_sarite += 1

X_train = np.array(X_list)
Y_train = np.array(Y_list)

X_train = X_train[..., np.newaxis]
X_max = X_train.max()
X_train = X_train.astype("float32") / X_max

label_mapping = {1: 0, 2: 1, 3: 2, 6: 3, 7: 4}
inverse_mapping = {v: k for k, v in label_mapping.items()}

Y_mapped = np.array([label_mapping[y] for y in Y_train])
NUM_CLASSES = 5
Y_train_OHE = to_categorical(Y_mapped, NUM_CLASSES)

X_train_split, X_val, Y_train_split, Y_val = train_test_split(
    X_train, Y_train_OHE, test_size=0.15, random_state=42
)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(Y_mapped),
    y=Y_mapped
)
class_weights = dict(enumerate(class_weights_array))

INPUT_SHAPE = (19, 19, 48, 1)

model = Sequential([
    Conv3D(32, (3, 3, 7), activation="relu", input_shape=INPUT_SHAPE),
    BatchNormalization(),
    MaxPooling3D((2, 2, 2)),
    Dropout(0.25),
    Conv3D(64, (3, 3, 5), activation="relu"),
    BatchNormalization(),
    MaxPooling3D((2, 2, 2)),
    Dropout(0.25),
    Conv3D(128, (3, 3, 3), activation="relu"),
    BatchNormalization(),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

EPOCHS = 5
BATCH_SIZE = 32

history = model.fit(
    X_train_split, Y_train_split,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, Y_val),
    class_weight=class_weights
)

Y_val_pred = model.predict(X_val)
Y_val_pred_labels = np.argmax(Y_val_pred, axis=1)
Y_val_true_labels = np.argmax(Y_val, axis=1)

macro_f1 = f1_score(Y_val_true_labels, Y_val_pred_labels, average="macro")
print(f"Macro F1 pe validare: {macro_f1:.4f}")

test_files = sorted([f for f in os.listdir(TEST_DATA_PATH) if f.endswith(".npy")])
X_test_list = []

for file_name in test_files:
    try:
        patch = np.load(os.path.join(TEST_DATA_PATH, file_name), allow_pickle=False)
        patch = patch.astype("float32") / X_max
        X_test_list.append(patch)
    except Exception:
        pass

X_test = np.array(X_test_list)[..., np.newaxis]

Y_test_proba = model.predict(X_test)
Y_test_labels = np.argmax(Y_test_proba, axis=1)
Y_test_final = [inverse_mapping[y] for y in Y_test_labels]

submission_df = pd.DataFrame({
    "filename": test_files,
    "label": Y_test_final
})

submission_df.to_csv("submission.csv", index=False)
print("submission.csv generat cu succes!")
