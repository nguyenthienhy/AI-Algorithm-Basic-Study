from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


def CNN_model(X_train , y_train , X_test , y_test , num_epochs , batch_size , num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

    checkpoint = ModelCheckpoint("keras_save_model.hdf5", monitor='loss', verbose=1,
                                 save_best_only=True, mode='auto', period=1)

    model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=num_epochs,
          verbose=1,
          validation_data=(X_test, y_test),
          callbacks=[checkpoint])
    
