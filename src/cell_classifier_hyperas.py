from keras.models import Sequential
from keras import layers, Model
from keras import Input
import numpy as np
from datetime import datetime
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.preprocessing.image import ImageDataGenerator


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
    test_images = dict_res["test_images"]
    test_labels = dict_res["test_labels"]
    input_shape = dict_res["input_shape"]

    return train_data, train_labels, valid_data, valid_labels, test_images, test_labels, input_shape


def build_hyperas_model(train_images, train_labels, valid_images, valid_labels, input_shape):
    input_tensor = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3))(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # If we choose 'three', add an additional third layer
    if ({{choice(['two', 'three'])}} == 'three'):
        x = layers.Conv2D(64, (3, 3))(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout({{uniform(0, 1)}})(x)

    x = layers.Dense({{choice([32, 64, 128])}})(x)  # used to be 64
    x = layers.BatchNormalization()(x)
    x = layers.Activation({{choice(['relu', 'tanh'])}})(x)

    output_tensor = layers.Dense(1, activation='sigmoid')(x)
    model = Model(input_tensor, output_tensor)

    model.compile(optimizer={{choice(['rmsprop', 'adam', 'sgd'])}},
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        fill_mode='constant',
        cval=0,
        # rescale=1,
        rotation_range=50,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        horizontal_flip=True
    )
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    train_datagen.fit(train_images)

    valid_datagen = ImageDataGenerator(rescale=1)

    batch_size = 64
    # fits the model on batches with real-time data augmentation:
    result = model.fit_generator(train_datagen.flow(train_images, train_labels, batch_size=batch_size,
                                                    shuffle=False),
                                 steps_per_epoch=len(train_images) / batch_size,
                                 epochs=30,
                                 shuffle=False,
                                 validation_steps=len(valid_images) / batch_size,
                                 validation_data=valid_datagen.flow(valid_images, valid_labels,
                                                                   batch_size=batch_size,
                                                                   shuffle=False))
    # result = model.fit(train_images, train_labels,
    #                    batch_size={{choice([32, 64])}},
    #                    epochs={{choice([5, 10, 20, 30])}},
    #                    validation_data=(valid_images, valid_labels))

    validation_acc = np.amax(result.history['val_acc'])
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def main():
    best_run, best_model = optim.minimize(model=build_hyperas_model,
                                          data=hyperas_data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    print("Evalutation of best performing model:")
    train_images, train_labels, valid_images, valid_labels, test_images, test_labels, input_shape = all_data()
    print(f"train_images {train_images.shape}, train_labels {train_labels.shape}, "
          f"valid_images {valid_images.shape}, valid_labels {valid_labels.shape}, "
          f"test_images {test_images.shape}, test_labels {test_labels.shape}")
    test_loss, test_acc = best_model.evaluate(test_images, test_labels)
    print(f"test_acc {test_acc}")
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    result_path = root_path + "results_classifier/"
    time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
    best_model.save(f'{result_path}cell_classifier_hyperas_{time_str}.h5')
    prediction = np.ndarray.flatten(best_model.predict(test_images))
    for i, predict_value in enumerate(prediction):
        predict_value = str(round(predict_value, 2))
        print(f"{i}: {predict_value} / {test_labels[i]}")
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    print(f"test_acc {test_acc}")

    best_model.save(f'{result_path}cell_classifier_model_acc_test_acc_{test_acc}_{time_str}.h5')
    best_model.save_weights(f'{result_path}cell_classifier_weights_acc_test_acc_{test_acc}_{time_str}.h5')
    # Save the model architecture
    with open(f'{result_path}cell_classifier_model_architecture_acc_test_acc_{test_acc}_{time_str}.json', 'w') as f:
        f.write(best_model.to_json())


main()
