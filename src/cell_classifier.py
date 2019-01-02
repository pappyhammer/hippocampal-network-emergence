import keras
from keras import models
from keras.models import Sequential, Model
from keras import layers
from keras import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from mouse_session_loader import load_mouse_sessions
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import pattern_discovery.tools.param as p_disc_tools_param
import time
from PIL import ImageSequence, ImageDraw
import PIL
from shapely import geometry
from matplotlib import pyplot as plt


class DataForMs(p_disc_tools_param.Parameters):
    def __init__(self, path_data, result_path):
        self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        super().__init__(path_results=result_path, time_str=self.time_str, bin_size=1)
        self.path_data = path_data
        self.cell_assemblies_data_path = None
        self.best_order_data_path = None

def load_movie(ms):
    if ms.tif_movie_file_name is not None:
        start_time = time.time()
        im = PIL.Image.open(ms.tif_movie_file_name)
        # im.show()
        # test = np.array(im)
        n_frames = len(list(ImageSequence.Iterator(im)))
        dim_x, dim_y = np.array(im).shape
        print(f"n_frames {n_frames}, dim_x {dim_x}, dim_y {dim_y}")
        ms.tiff_movie = np.zeros((n_frames, dim_x, dim_y), dtype="uint16")
        for frame, page in enumerate(ImageSequence.Iterator(im)):
            ms.tiff_movie[frame] = np.array(page)
        stop_time = time.time()
        print(f"Time for loading movie: "
              f"{np.round(stop_time-start_time, 3)} s")
        return True
    return False

def scale_polygon_to_source(poly_gon, minx, miny):
    coords = list(poly_gon.exterior.coords)
    scaled_coords = []
    for coord in coords:
        scaled_coords.append((coord[0] - minx, coord[1] - miny))
    return geometry.Polygon(scaled_coords)

def get_source_profile(cell, ms, binary_version=True, pixels_around=0, bounds=None, buffer=None):
    # print("get_source_profile")
    len_frame_x = ms.tiff_movie[0].shape[1]
    len_frame_y = ms.tiff_movie[0].shape[0]

    # determining the size of the square surrounding the cell
    poly_gon = ms.coord_obj.cells_polygon[cell]
    if bounds is None:
        minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
    else:
        minx, miny, maxx, maxy = bounds

    minx = max(0, minx - pixels_around)
    miny = max(0, miny - pixels_around)
    maxx = min(len_frame_x - 1, maxx + pixels_around)
    maxy = min(len_frame_y - 1, maxy + pixels_around)

    len_x = maxx - minx + 1
    len_y = maxy - miny + 1

    # mask used in order to keep only the cells pixel
    # the mask put all pixels in the polygon, including the pixels on the exterior line to zero
    scaled_poly_gon = scale_polygon_to_source(poly_gon=poly_gon, minx=minx, miny=miny)
    img = PIL.Image.new('1', (len_x, len_y), 1)
    if buffer is not None:
        scaled_poly_gon = scaled_poly_gon.buffer(buffer)
    ImageDraw.Draw(img).polygon(list(scaled_poly_gon.exterior.coords), outline=0, fill=0)
    mask = np.array(img)
    # mask = np.ones((len_x, len_y))
    # cv2.fillPoly(mask, scaled_poly_gon, 0)
    # mask = mask.astype(bool)

    if binary_version:
        source_profile = np.ones((len_y, len_x))
    else:
        source_profile = np.zeros((len_y, len_x))

    if not binary_version:
        # selectionning the best peak to produce the source_profile
        peaks = np.where(ms.spike_struct.peak_nums[cell, :] > 0)[0]
        threshold = np.percentile(ms.traces[cell, peaks], 95)
        selected_peaks = peaks[np.where(ms.traces[cell, peaks] > threshold)[0]]
        # max 10 peaks, min 5 peaks
        if len(selected_peaks) > 10:
            p = 10 / len(peaks)
            threshold = np.percentile(ms.traces[cell, peaks], (1 - p) * 100)
            selected_peaks = peaks[np.where(ms.traces[cell, peaks] > threshold)[0]]
        elif (len(selected_peaks) < 5) and (len(peaks) > 5):
            p = 5 / len(peaks)
            threshold = np.percentile(ms.traces[cell, peaks], (1 - p) * 100)
            selected_peaks = peaks[np.where(ms.traces[cell, peaks] > threshold)[0]]

        # print(f"threshold {threshold}")
        # print(f"n peaks: {len(selected_peaks)}")

        onsets_frames = np.where(ms.spike_struct.spike_nums[cell, :] > 0)[0]
        raw_traces = np.copy(ms.raw_traces)
        # so the lowest value is zero
        raw_traces += abs(np.min(raw_traces))
        for peak in selected_peaks:
            tmp_source_profile = np.zeros((len_y, len_x))
            onsets_before_peak = np.where(onsets_frames <= peak)[0]
            if len(onsets_before_peak) == 0:
                # shouldn't arrive
                continue
            onset = onsets_frames[onsets_before_peak[-1]]
            # print(f"onset {onset}, peak {peak}")
            frames_tiff = ms.tiff_movie[onset:peak + 1]
            for frame_index, frame_tiff in enumerate(frames_tiff):
                tmp_source_profile += (frame_tiff[miny:maxy + 1, minx:maxx + 1] * raw_traces[cell, onset + frame_index])
            # averaging
            tmp_source_profile = tmp_source_profile / (np.sum(raw_traces[cell, onset:peak + 1]))
            source_profile += tmp_source_profile

        source_profile = source_profile / len(selected_peaks)

    return source_profile, minx, miny, mask

def load_data(ms_to_use, param, use_binary_mask=True, split_values=(0.5, 0.3)):
    # ms_to_use: list of string representing the mouse_session
    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=False, load_abf=False,
                                            for_cell_classifier=True)

    # first we determine the max dimension of the array that will contains the cell contours
    max_width = 0
    max_height = 0
    padding_value = 10
    total_n_cells = 0
    for ms in ms_str_to_ms_dict.values():
        movie_loaded = load_movie(ms)
        if not movie_loaded:
            raise Exception(f"could not load movie of ms {ms.description}")
        n_cells = ms.spike_struct.n_cells
        total_n_cells += n_cells
        for cell in np.arange(n_cells):
            poly_gon = ms.coord_obj.cells_polygon[cell]
            minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
            max_width = max(max_width, maxx-minx)
            max_height = max(max_height, maxy-miny)
    # then we add a few pixels
    max_width += padding_value
    max_height += padding_value

    # data will be 0 or 1
    if use_binary_mask:
        full_data = np.zeros((total_n_cells, max_height, max_width), dtype="uint8")
    else:
        full_data = np.zeros((total_n_cells, max_height, max_width))
    full_labels = np.zeros(total_n_cells, dtype="uint8")

    cells_count = 0
    for ms in ms_str_to_ms_dict.values():
        n_cells = ms.spike_struct.n_cells

        labels = np.ones(n_cells, dtype="uint8")
        labels[ms.cells_to_remove] = 0
        full_labels[cells_count:cells_count+n_cells] = labels

        # for each cell, we will extract a 2D array representing the cell shape
        # all 2D array should have the same shape
        for cell in np.arange(n_cells):
            source_profile, minx, miny, mask_source_profile = get_source_profile(ms=ms, binary_version=True,
                                                                                 cell=cell,
                                                                                 pixels_around=0,
                                                                                 bounds=None)
            source_profile[mask_source_profile] = 0
            profile_fit = np.zeros((max_height, max_width), dtype="uint8")
            # we center the source profile
            y_coord = (profile_fit.shape[0] - source_profile.shape[0]) // 2
            x_coord = (profile_fit.shape[1] - source_profile.shape[1]) // 2
            profile_fit[y_coord:source_profile.shape[0]+y_coord, x_coord:source_profile.shape[1]+x_coord] = \
                source_profile

            visualize_cells = False
            if visualize_cells:
                fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                        gridspec_kw={'height_ratios': [1],
                                                     'width_ratios': [1]},
                                        figsize=(5, 5))
                c_map = plt.get_cmap('gray')
                img_src_profile = ax1.imshow(profile_fit, cmap=c_map)
                plt.show()

            full_data[cells_count+cell] = profile_fit


        cells_count += n_cells

    print(f"total_n_cells {total_n_cells}")
    print(f"full_labels {len(np.where(full_labels)[0])}")

    # 0.3 of the data will be for the test, the rest for training
    # train_data, test_data, train_labels, test_labels = train_test_split(full_data, full_labels, train_size=.7)

    # cells shuffling
    cells_shuffling = np.arange(total_n_cells)
    np.random.shuffle(cells_shuffling)
    n_cells_for_training = int(total_n_cells * split_values[0])
    n_cells_for_validation = int(total_n_cells * split_values[1])

    train_data = full_data[cells_shuffling[:n_cells_for_training]]
    train_labels = full_labels[cells_shuffling[:n_cells_for_training]]
    valid_data = full_data[cells_shuffling[n_cells_for_training:n_cells_for_training+n_cells_for_validation]]
    valid_labels = full_labels[cells_shuffling[n_cells_for_training:n_cells_for_training+n_cells_for_validation]]
    test_data = full_data[cells_shuffling[n_cells_for_training+n_cells_for_validation:]]
    test_labels = full_labels[cells_shuffling[n_cells_for_training+n_cells_for_validation:]]

    return (train_data, train_labels), (valid_data, valid_labels), (test_data, test_labels)


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def build_model(input_shape):
    use_sequential_mode = False
    if use_sequential_mode:
        model = Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # print(model.summary())
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        input_tensor = Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_tensor)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        output_tensor = layers.Dense(1, activation='sigmoid')(x)

        model = Model(input_tensor, output_tensor)

    return model


def plot_training_and_validation_loss(history, n_epochs):
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, n_epochs + 1)
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                            gridspec_kw={'height_ratios': [1],
                                         'width_ratios': [1]},
                            figsize=(5, 5))
    ax1.plot(epochs, smooth_curve(loss_values), 'bo', label='Training loss')
    ax1.plot(epochs, smooth_curve(val_loss_values), 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.close()


def plot_training_and_validation_accuracy(history, n_epochs):
    history_dict = history.history
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    epochs = range(1, n_epochs + 1)
    plt.plot(epochs, smooth_curve(acc_values), 'bo', label='Training acc')
    plt.plot(epochs, smooth_curve(val_acc_values), 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()
    plt.close()

def train_model():
    root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
    path_data = root_path + "data/"
    result_path = root_path + "results_classifier/"
    binary_version = True

    param = DataForMs(path_data=path_data, result_path=result_path)
    (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels)\
        = load_data(["p12_171110_a000_ms"], param=param, use_binary_mask=binary_version,
                    split_values=(0.5, 0.3))

    print(f"train_images {train_images.shape}, train_labels {train_labels.shape}, "
          f"valid_images {valid_images.shape}, valid_labels {valid_labels.shape}, "
          f"test_images {test_images.shape}, test_labels {test_labels.shape}")

    train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
    if not binary_version:
        train_images = train_images.astype('float32') / 255

    valid_images = valid_images.reshape((valid_images.shape[0], valid_images.shape[1], valid_images.shape[2], 1))
    if not binary_version:
        valid_images = valid_images.astype('float32') / 255

    test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
    if not binary_version:
        test_images = test_images.astype('float32') / 255

    print(f"train_images {train_images.shape}, train_labels {train_labels.shape}, "
          f"valid_images {valid_images.shape}, valid_labels {valid_labels.shape}, "
          f"test_images {test_images.shape}, test_labels {test_labels.shape}")

    train_datagen = ImageDataGenerator(
        fill_mode='constant',
        cval=0,
        # rescale=1,
        rotation_range=120,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True
    )

    model = build_model(input_shape=train_images.shape[1:])
    print(model.summary())
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    n_epochs = 30
    batch_size = 64

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    train_datagen.fit(train_images)

    # fits the model on batches with real-time data augmentation:
    history = model.fit_generator(train_datagen.flow(train_images, train_labels, batch_size=batch_size),
                        steps_per_epoch=len(train_images) / batch_size, epochs=n_epochs,
                        validation_data=(valid_images, valid_labels))
    # model.fit(train_images, train_labels, epochs=n_epochs, batch_size=64)
    # history = model.fit(train_images,
    #                     train_labels,
    #                     epochs=n_epochs,
    #                     batch_size=batch_size,
    #                     validation_data=(valid_images, valid_labels))

    model.save(f'{param.path_results}cell_classifier_{param.time_str}.h5')

    show_plots = True

    if show_plots:
        plot_training_and_validation_loss(history, n_epochs)
        plot_training_and_validation_accuracy(history, n_epochs)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"test_acc {test_acc}")

    # then to predict: model.predict(x_test)

train_model()