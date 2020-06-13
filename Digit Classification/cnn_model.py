from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from PIL import Image
import numpy as np
import constant

def CNN_model(X_train, y_train, X_test, y_test, num_epochs, batch_size, num_classes):
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # checkpoint
    filepath = "saved_model/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model.fit(X_train, y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list
    )


def load_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(36, activation='softmax'))
    model.load_weights("saved_model/weights.best.hdf5")
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def predict(model, path):
    # filename = process_image.convert_to_white_background(path)
    # show image of test
    image = Image.open(path).convert('L')
    array = np.array(image.resize((28, 28)), dtype=np.float32)
    array = array.reshape(28 * 28, 1) / 255.0
    array = np.array([array])
    array = array.reshape(1, 28, 28, 1).astype('float32')
    return constant.defines[model.predict_classes(array)[0]]

def predictOutNum(model, path):
    # filename = process_image.convert_to_white_background(path)
    # show image of test
    image = Image.open(path).convert('L')
    array = np.array(image.resize((28, 28)), dtype=np.float32)
    array = array.reshape(28 * 28, 1) / 255.0
    array = np.array([array])
    array = array.reshape(1, 28, 28, 1).astype('float32')
    return model.predict_classes(array)[0]


