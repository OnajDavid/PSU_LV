from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_s = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test_s = x_test.reshape(-1, 28, 28, 1) / 255.0

y_train_s = to_categorical(y_train, num_classes=10)
y_test_s = to_categorical(y_test, num_classes=10)


# TODO: strukturiraj konvolucijsku neuronsku mrezu
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.GlobalAveragePooling2D(),
    layers.Dense(10, activation='softmax')
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
    x_train_s, y_train_s,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[tensorboard_callback, model_checkpoint]
)

#TODO: Ucitaj najbolji model
best_model = keras.models.load_model(checkpoint_filepath)


# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_preds = best_model.predict(x_train_s)
test_preds = best_model.predict(x_test_s)

train_labels = np.argmax(train_preds, axis=1)
test_labels = np.argmax(test_preds, axis=1)

train_accuracy = accuracy_score(y_train, train_labels)
test_accuracy = accuracy_score(y_test, test_labels)

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

plot_confusion_matrix(y_train, train_labels, "Matrica zabune - Trening skup")
plot_confusion_matrix(y_test, test_labels, "Matrica zabune - Test skup")
