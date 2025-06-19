from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

train_ds = image_dataset_from_directory(
    directory='C:\\Users\\student\\Downloads\\gtsrb\\Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    subset="training",
    seed=123,
    validation_split=0.2,
    image_size=(48, 48)
)
validation_ds = image_dataset_from_directory(
directory='C:\\Users\\student\\Downloads\\gtsrb\\Train',
labels='inferred',
label_mode='categorical',
batch_size=32,
subset="validation",
seed=123,
validation_split=0.2,
image_size=(48, 48)
)
test_ds = image_dataset_from_directory(
directory='C:\\Users\\student\\Downloads\\gtsrb\\Test',
labels='inferred',
label_mode='categorical',
batch_size=32,
image_size=(48, 48)
)
# TODO: strukturiraj konvolucijsku neuronsku mrezu
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(43, activation='softmax')
])

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# TODO: definiraj callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
checkpoint_filepath = 'best_model.h5'
model_checkpoint = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# TODO: provedi treniranje mreze pomocu .fit()
history = model.fit(
    train_ds, validation_data=validation_ds,
    epochs=10,
    batch_size=64,
    callbacks=[tensorboard_callback, model_checkpoint]
)

#TODO: Ucitaj najbolji model
best_model = keras.models.load_model(checkpoint_filepath)


# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_preds = best_model.predict(train_ds)
test_preds = best_model.predict(test_ds)

train_pred_labels = np.argmax(train_preds, axis=1)
test_pred_labels = np.argmax(test_preds, axis=1)

train_true_labels = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in train_ds])
test_true_labels = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])

train_accuracy = accuracy_score(train_true_labels, train_pred_labels)
test_accuracy = accuracy_score(test_true_labels, test_pred_labels)

print(f"Točnost na trening skupu: {train_accuracy:.4f}")
print(f"Točnost na test skupu: {test_accuracy:.4f}")



# TODO: Prikazite matricu zabune na skupu podataka za testiranje
def plot_confusion_matrix(true_labels, pred_labels, title):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

plot_confusion_matrix(train_true_labels, train_pred_labels, "Matrica zabune - Trening skup")
plot_confusion_matrix(test_true_labels, test_pred_labels, "Matrica zabune - Test skup")
