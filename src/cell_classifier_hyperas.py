from keras.models import Sequential
from keras import layers, Model
from keras import Input
import numpy as np
from datetime import datetime
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

def hyperas_data():
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    result_path = root_path + "results_classifier/"
    dict_res = np.load(result_path + "data_hyperas.npz")

    train_images = dict_res["train_images"]
    train_labels = dict_res["train_labels"]
    valid_images = dict_res["valid_images"]
    valid_labels = dict_res["valid_labels"]
    input_shape = dict_res["input_shape"]

    return train_images, train_labels, valid_images, valid_labels, input_shape

def all_data():
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    result_path = root_path + "results_classifier/"
    dict_res = np.load(result_path + "data_hyperas.npz")

    train_data = dict_res["train_images"]
    train_labels = dict_res["train_labels"]
    valid_data = dict_res["valid_images"]
    valid_labels = dict_res["valid_labels"]
    test_images = dict_res["valid_images"]
    test_labels = dict_res["valid_labels"]
    input_shape = dict_res["input_shape"]

    return train_data, train_labels, valid_data, valid_labels, test_images, test_labels, input_shape


def build_hyperas_model(train_images, train_labels, valid_images, valid_labels, input_shape):
    input_tensor = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3))(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # If we choose 'three', add an additional third layer
    if ({{choice(['two', 'three'])}} == 'three'):
        x = layers.Conv2D(64, (3, 3))(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dropout({{uniform(0, 1)}})(x)

    x = layers.Dense({{choice([32, 64, 128])}})(x)  # used to be 64

    x = layers.Activation({{choice(['relu', 'tanh'])}})(x)

    output_tensor = layers.Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output_tensor)

    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    result = model.fit(train_images, train_labels,
                       batch_size={{choice([32, 64])}},
                       epochs={{choice([5, 10, 20, 30])}},
                       validation_data=(valid_images, valid_labels))
    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def main():
    best_run, best_model = optim.minimize(model=build_hyperas_model,
                                          data=hyperas_data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    print("Evalutation of best performing model:")
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels, input_shape = all_data()
    print(best_model.evaluate(test_images, test_labels))
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    result_path = root_path + "results_classifier/"
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    best_model.save(f'{result_path}cell_classifier_hyperas_{time_str}.h5')
    prediction = np.ndarray.flatten(best_model.predict(test_images))
    for i, predict_value in enumerate(prediction):
        predict_value = str(round(predict_value, 2))
        print(f"test predict / real: {predict_value} / {test_labels[i]}")
    print("Best performing model chosen hyper-parameters:")
    print(best_run)

main()
