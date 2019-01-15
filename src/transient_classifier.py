import numpy as np
import hdf5storage
from keras.layers import Conv2D, MaxPooling2D, Flatten, Bidirectional
from keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras import layers
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import pattern_discovery.tools.param as p_disc_tools_param
from datetime import datetime
import time
from PIL import ImageSequence, ImageDraw
import PIL
from mouse_session_loader import load_mouse_sessions
from shapely import geometry
from scipy import ndimage

import sys
import platform
print(f"sys.maxsize {sys.maxsize}, platform.architecture {platform.architecture()}")


# make a function to build the training and validation set (test)
# same for the training labels and test labels
# function equivalent to
# (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# with data being vectorizes and labels adequates

# one way will be to split the data using a sliding window with overlap of like 50%, and we label
# the frame during which the cell is active

# another way we give him all the potential transients and he learn which on are true and which ones are false.
# but then it won't learn to fusion some onset/peaks periods for long transient

# we will need a generic fucntion that will take a mouse_session in argument and the matrices containing the onsets
# and peaks, and will return a bunch of data and labels. Then we will concatenate the results from different session
# and then split it between, train and test data

# we need to save the weights

# TimeDistributed: https://keras.io/layers/wrappers/ with BiDirectionnal wrapper just below
# exemple: https://keras.io/getting-started/functional-api-guide/

class DataForMs(p_disc_tools_param.Parameters):
    def __init__(self, path_data, result_path):
        self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        super().__init__(path_results=result_path, time_str=self.time_str, bin_size=1)
        self.path_data = path_data
        self.cell_assemblies_data_path = None
        self.best_order_data_path = None


def load_movie(ms):
    if ms.tif_movie_file_name is not None:
        if ms.tiff_movie is None:
            start_time = time.time()
            im = PIL.Image.open(ms.tif_movie_file_name)
            n_frames = len(list(ImageSequence.Iterator(im)))
            dim_x, dim_y = np.array(im).shape
            print(f"n_frames {n_frames}, dim_x {dim_x}, dim_y {dim_y}")
            ms.tiff_movie = np.zeros((n_frames, dim_x, dim_y))
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


def get_source_profile_frames(cell, ms, frames, pixels_around=3, buffer=None):
    len_frame_x = ms.tiff_movie[0].shape[1]
    len_frame_y = ms.tiff_movie[0].shape[0]

    # determining the size of the square surrounding the cell so it includes all overlapping cells around
    overlapping_cells = ms.coord_obj.intersect_cells[cell]
    cells_to_display = [cell]
    cells_to_display.extend(overlapping_cells)
    poly_gon = ms.coord_obj.cells_polygon[cell]

    # calculating the bound that will surround all the cells
    minx = None
    maxx = None
    miny = None
    maxy = None

    for cell_to_display in cells_to_display:
        poly_gon = ms.coord_obj.cells_polygon[cell_to_display]

        if minx is None:
            minx, miny, maxx, maxy = np.array(list(poly_gon.bounds)).astype(int)
        else:
            tmp_minx, tmp_miny, tmp_maxx, tmp_maxy = np.array(list(poly_gon.bounds)).astype(int)
            minx = min(minx, tmp_minx)
            miny = min(miny, tmp_miny)
            maxx = max(maxx, tmp_maxx)
            maxy = max(maxy, tmp_maxy)

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

    # source_profile = np.zeros((len(frames), len_y, len_x))

    frames_tiff = ms.tiff_movie[frames]
    source_profile = frames_tiff[:, miny:maxy + 1, minx:maxx + 1]

    # normalized so that value are between 0 and 1
    source_profile = source_profile / np.max(ms.tiff_movie)

    return source_profile, mask


def load_data(param, split_values=(0.6, 0.2), with_border=False, sliding_window_len=100,
              movies_shuffling=None, with_shuffling=True,
              overlap_value=0.5):
    # TODO: option to remove overlapp
    ms_to_use = ["p12_171110_a000_ms", "p7_171012_a000_ms"]
    ms_str_to_ms_dict = load_mouse_sessions(ms_str_to_load=ms_to_use,
                                            param=param,
                                            load_traces=False, load_abf=False,
                                            for_transient_classifier=True)
    cell_to_load_by_ms = {"p12_171110_a000_ms": np.arange(5), "p7_171012_a000_ms": np.arange(70)}

    total_n_cells = 0
    n_movies = 0

    # used to check the result on original data, useful after shuffling
    movies_descr = []

    # loading movies
    # and counting cells and movies
    for ms_str in ms_to_use:
        ms = ms_str_to_ms_dict[ms_str]
        cells_to_load_tmp = cell_to_load_by_ms[ms_str]
        cells_to_load = []
        for cell in cells_to_load_tmp:
            if cell not in ms.cells_to_remove:
                cells_to_load.append(cell)
        total_n_cells += len(cells_to_load)
        cells_to_load = np.array(cells_to_load)
        cell_to_load_by_ms[ms_str] = cells_to_load
        n_frames = ms.spike_struct.spike_nums_dur.shape[1]
        n_movies += int(np.ceil(n_frames / (sliding_window_len * overlap_value))) - 1

        movie_loaded = load_movie(ms)
        if not movie_loaded:
            raise Exception(f"could not load movie of ms {ms.description}")

    max_width = 30
    max_height = 30
    n_movies = n_movies * total_n_cells
    print(f"n_movies {n_movies}")

    full_data = np.zeros((n_movies, sliding_window_len, max_height, max_width))
    full_data_masked = np.zeros((n_movies, sliding_window_len, max_height, max_width))
    # for each movie, each frame is 1 if the cell is active, 0 if not active
    full_labels = np.zeros((n_movies, sliding_window_len), dtype="uint8")

    movie_count = 0
    for ms_str in ms_to_use:
        ms = ms_str_to_ms_dict[ms_str]
        spike_nums_dur = ms.spike_struct.spike_nums_dur
        n_frames = spike_nums_dur.shape[1]
        for cell in cell_to_load_by_ms[ms_str]:
            # then we slide the window
            # frames index of the beginning of each movie
            indices_movies = np.arange(0, n_frames, int(sliding_window_len * overlap_value))
            # print(f"len(indices_movies) {len(indices_movies)}")
            for i, index_movie in enumerate(indices_movies):
                if i == (len(indices_movies) - 1):
                    if (indices_movies[i - 1] + sliding_window_len) == n_frames:
                        break
                    else:
                        frames = np.arange(n_frames - sliding_window_len, n_frames)
                else:
                    frames = np.arange(index_movie, index_movie + sliding_window_len)
                # setting labels: active frame or not
                full_labels[movie_count] = spike_nums_dur[cell, frames]
                # now adding the movie of those frames in this slinding_window
                source_profile_frames, mask_source_profile = get_source_profile_frames(cell=cell, ms=ms,
                                                                                       frames=frames, pixels_around=3,
                                                                                       buffer=None)
                # if i == 0:
                #     print(f"source_profile_frames.shape {source_profile_frames.shape}")
                source_profile_frames_masked = np.copy(source_profile_frames)
                source_profile_frames_masked[:, mask_source_profile] = 0

                profile_fit = np.zeros((len(frames), max_height, max_width))
                profile_fit_masked = np.zeros((len(frames), max_height, max_width))
                # we center the source profile
                y_coord = (profile_fit.shape[1] - source_profile_frames.shape[1]) // 2
                x_coord = (profile_fit.shape[2] - source_profile_frames.shape[2]) // 2
                profile_fit[:, y_coord:source_profile_frames.shape[1] + y_coord,
                x_coord:source_profile_frames.shape[2] + x_coord] = \
                    source_profile_frames
                profile_fit_masked[:, y_coord:source_profile_frames.shape[1] + y_coord,
                x_coord:source_profile_frames.shape[2] + x_coord] = \
                    source_profile_frames_masked

                full_data[movie_count] = profile_fit
                full_data_masked[movie_count] = profile_fit_masked

                visualize_cells = False
                if visualize_cells:
                    for i_img, img in enumerate(profile_fit):
                        # print(f"max value {np.max(source_profile)}")
                        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                                gridspec_kw={'height_ratios': [1],
                                                             'width_ratios': [1]},
                                                figsize=(5, 5))
                        c_map = plt.get_cmap('gray')
                        img_profile = ax1.imshow(img, cmap=c_map)
                        plt.show()
                        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                                gridspec_kw={'height_ratios': [1],
                                                             'width_ratios': [1]},
                                                figsize=(5, 5))
                        c_map = plt.get_cmap('gray')
                        img_profile = ax1.imshow(profile_fit_masked[i_img], cmap=c_map)
                        plt.show()
                movies_descr.append(f"{ms.description}_cell_{cell}_first_frame_{index_movie}")
                movie_count += 1
    print(f"movie_count {movie_count}")
    # cells shuffling
    if movies_shuffling is None:
        movies_shuffling = np.arange(n_movies)
        if with_shuffling:
            np.random.shuffle(movies_shuffling)
    n_movies_for_training = int(n_movies * split_values[0])
    n_movies_for_validation = int(n_movies * split_values[1])

    train_data = full_data[movies_shuffling[:n_movies_for_training]]
    train_data_masked = full_data_masked[movies_shuffling[:n_movies_for_training]]
    train_labels = full_labels[movies_shuffling[:n_movies_for_training]]

    # data augmentation, we rotate each movie 3 times
    # we could also translate the movie a few pixels left or right
    do_data_augmentation = False
    if do_data_augmentation:
        train_data_augmented = np.zeros((train_data.shape[0] * 4, sliding_window_len, max_height, max_width))
        train_data_masked_augmented = np.zeros((train_data.shape[0] * 4, sliding_window_len, max_height, max_width))
        train_labels_augmented = np.zeros((train_labels.shape[0]*4, sliding_window_len), dtype="uint8")
        for index_movie in np.arange(train_data.shape[0]):
            train_data_augmented[index_movie * 4] = train_data[index_movie]
            train_data_masked_augmented[index_movie * 4] = train_data_masked[index_movie]
            train_labels_augmented[index_movie * 4] = train_labels[index_movie]
            for i_angle, angle in enumerate([90, 180, 270]):
                first_index = (index_movie * 4) + i_angle + 1
                # looping on each frame
                for sl_win_i in np.arange(train_data.shape[1]):
                    to_rotate = train_data[index_movie, sl_win_i]
                    train_data_augmented[first_index, sl_win_i] = ndimage.rotate(input=to_rotate, angle=angle,
                                                                                 reshape=False)
                    to_rotate = train_data_masked[index_movie, sl_win_i]
                    train_data_masked_augmented[first_index, sl_win_i] = ndimage.rotate(input=to_rotate, angle=angle,
                                                                                        reshape=False)

                train_labels_augmented[first_index] = train_labels[index_movie]
            visualize_cells = False
            if visualize_cells and (index_movie == 0):
                root_path = "/Users/pappyhammer/Documents/academique/these_inmed/robin_michel_data/"
                path_data = root_path + "data/"
                result_path = root_path + "results_classifier/"

                images = [train_data_augmented[0, 0], train_data_augmented[1, 0], train_data_augmented[2, 0],
                          train_data_augmented[3, 0]]
                images_masked = [train_data_masked_augmented[0, 0], train_data_masked_augmented[1, 0],
                                 train_data_masked_augmented[2, 0], train_data_masked_augmented[3, 0]]
                for i_img in np.arange(len(images)):
                    # print(f"max value {np.max(source_profile)}")
                    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                            gridspec_kw={'height_ratios': [1],
                                                         'width_ratios': [1]},
                                            figsize=(5, 5))
                    c_map = plt.get_cmap('gray')
                    img_profile = ax1.imshow(images[i_img], cmap=c_map)
                    fig.savefig(f'{result_path}/cell_{i_img}.pdf',
                                format=f"pdf")
                    # plt.show()
                    plt.close()
                    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                            gridspec_kw={'height_ratios': [1],
                                                         'width_ratios': [1]},
                                            figsize=(5, 5))
                    c_map = plt.get_cmap('gray')
                    img_profile = ax1.imshow(images_masked[i_img], cmap=c_map)
                    fig.savefig(f'{result_path}/cell_masked_{i_img}.pdf',
                                format=f"pdf")
                    # plt.show()
                    plt.close()
        train_data = train_data_augmented
        train_data_masked = train_data_masked_augmented
        train_labels = train_labels_augmented

    valid_data = full_data[movies_shuffling[n_movies_for_training:n_movies_for_training + n_movies_for_validation]]
    valid_data_masked = full_data_masked[movies_shuffling[n_movies_for_training:
                                                          n_movies_for_training + n_movies_for_validation]]
    valid_labels = full_labels[movies_shuffling[n_movies_for_training:n_movies_for_training + n_movies_for_validation]]
    test_data = full_data[movies_shuffling[n_movies_for_training + n_movies_for_validation:]]
    test_data_masked = full_data_masked[movies_shuffling[n_movies_for_training + n_movies_for_validation:]]
    test_labels = full_labels[movies_shuffling[n_movies_for_training + n_movies_for_validation:]]
    # print(f"test cells: {cells_shuffling[n_cells_for_training + n_movies_for_validation:]}")
    test_movie_descr = []
    for movie in movies_shuffling[n_movies_for_training + n_movies_for_validation:]:
        test_movie_descr.append(movies_descr[movie])

    data = (train_data, valid_data, test_data)
    data_masked = (train_data_masked, valid_data_masked, test_data_masked)
    labels = (train_labels, valid_labels, test_labels)

    return data, data_masked, labels, test_movie_descr


def build_model(input_shape, use_mulimodal_inputs=False, dropout_value=0):
    n_frames = input_shape[0]
    # First, let's define a vision model using a Sequential model.
    # This model will encode an image into a vector.
    vision_model = Sequential()
    vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape[1:]))
    vision_model.add(Conv2D(64, (3, 3), activation='relu'))
    vision_model.add(MaxPooling2D((2, 2)))
    # vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # vision_model.add(Conv2D(128, (3, 3), activation='relu'))
    # vision_model.add(MaxPooling2D((2, 2)))
    if dropout_value > 0:
        vision_model.add(layers.Dropout(dropout_value))
    # vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    # vision_model.add(Conv2D(256, (3, 3), activation='relu'))
    # vision_model.add(Conv2D(256, (3, 3), activation='relu'))
    # vision_model.add(MaxPooling2D((2, 2)))
    vision_model.add(Flatten())

    video_input = Input(shape=input_shape, name="video_input")
    # This is our video encoded via the previously trained vision_model (weights are reused)
    encoded_frame_sequence = TimeDistributed(vision_model)(video_input)  # the output will be a sequence of vectors
    # encoded_video = LSTM(256)(encoded_frame_sequence)  # the output will be a vector
    encoded_video = Bidirectional(LSTM(128, return_sequences=True))(encoded_frame_sequence)
    # if we put input_shape in Bidirectional, it crashes
    # encoded_video = Bidirectional(LSTM(128, return_sequences=True),
    #                               input_shape=(n_frames, 128))(encoded_frame_sequence)
    encoded_video = Bidirectional(LSTM(256))(encoded_video)

    video_input_masked = Input(shape=input_shape, name="video_input_masked")
    # This is our video encoded via the previously trained vision_model (weights are reused)
    encoded_frame_sequence_masked = TimeDistributed(vision_model)(
        video_input_masked)  # the output will be a sequence of vectors
    # encoded_video_masked = LSTM(256)(encoded_frame_sequence_masked)  # the output will be a vector

    encoded_video_masked = Bidirectional(LSTM(128, return_sequences=True))(encoded_frame_sequence_masked)
    encoded_video_masked = Bidirectional(LSTM(256))(encoded_video_masked)

    # in case we want 2 videos, one with masked, and one with the cell centered
    # And this is our video question answering model:
    if use_mulimodal_inputs:
        merged = layers.concatenate([encoded_video, encoded_video_masked])
        # output = TimeDistributed(Dense(1, activation='sigmoid')))
        output = Dense(100, activation='sigmoid')(merged)
        video_model = Model(inputs=[video_input, video_input_masked], outputs=output)
    else:
        # output = TimeDistributed(Dense(1, activation='sigmoid'))(encoded_video)
        output = Dense(100, activation='sigmoid')(encoded_video)
        video_model = Model(inputs=video_input, outputs=output)

    return video_model


def get_source_profile_for_prediction(ms, cell, max_width=30, max_height=30, sliding_window_len=100):
    n_frames = len(ms.tiff_movie)
    count_is_good = True
    if (n_frames % sliding_window_len) == 0:
        n_movies = n_frames // sliding_window_len
    else:
        n_movies = (n_frames // sliding_window_len) + 1
        count_is_good = False

    full_data = np.zeros((n_movies, sliding_window_len, max_height, max_width))
    full_data_masked = np.zeros((n_movies, sliding_window_len, max_height, max_width))

    movie_count = 0
    for index_movie in np.arange(n_movies):
        if (index_movie == (n_movies - 1)) and (not count_is_good):
            # last part is not equal to sliding_window_len
            # there will be some overlap with the last one
            frames = np.arange(n_frames - sliding_window_len, n_frames)
        else:
            frames = np.arange(index_movie * sliding_window_len, (index_movie + 1) * sliding_window_len)
        source_profile_frames, mask_source_profile = get_source_profile_frames(cell=cell, ms=ms,
                                                                               frames=frames, pixels_around=3,
                                                                               buffer=None)
        # if i == 0:
        #     print(f"source_profile_frames.shape {source_profile_frames.shape}")
        source_profile_frames_masked = np.copy(source_profile_frames)
        source_profile_frames_masked[:, mask_source_profile] = 0

        profile_fit = np.zeros((len(frames), max_height, max_width))
        profile_fit_masked = np.zeros((len(frames), max_height, max_width))
        # we center the source profile
        y_coord = (profile_fit.shape[1] - source_profile_frames.shape[1]) // 2
        x_coord = (profile_fit.shape[2] - source_profile_frames.shape[2]) // 2
        profile_fit[:, y_coord:source_profile_frames.shape[1] + y_coord,
        x_coord:source_profile_frames.shape[2] + x_coord] = \
            source_profile_frames
        profile_fit_masked[:, y_coord:source_profile_frames.shape[1] + y_coord,
        x_coord:source_profile_frames.shape[2] + x_coord] = \
            source_profile_frames_masked

        full_data[movie_count] = profile_fit
        full_data_masked[movie_count] = profile_fit_masked
        movie_count += 1

    return full_data, full_data_masked


def predict_transient_from_saved_model(ms, cell, weights_file, json_file):
    start_time = time.time()
    # Model reconstruction from JSON file
    with open(json_file, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(weights_file)
    stop_time = time.time()
    print(f"Time for loading model: "
          f"{np.round(stop_time-start_time, 3)} s")

    start_time = time.time()
    n_frames = len(ms.tiff_movie)
    multi_inputs = (model.layers[0].output_shape == model.layers[1].output_shape)
    sliding_window_len = model.layers[0].output_shape[1]
    max_height = model.layers[0].output_shape[2]
    max_width = model.layers[0].output_shape[3]
    data, data_masked = get_source_profile_for_prediction(ms=ms, cell=cell,
                                                          sliding_window_len=sliding_window_len,
                                                          max_width=max_width, max_height=max_height)
    data = data.reshape((data.shape[0], data.shape[1], data.shape[2],
                         data.shape[3], 1))
    data_masked = data_masked.reshape((data_masked.shape[0], data_masked.shape[1], data_masked.shape[2],
                                       data_masked.shape[3], 1))
    stop_time = time.time()
    print(f"Time to get the data: "
          f"{np.round(stop_time-start_time, 3)} s")

    start_time = time.time()
    if multi_inputs:
        predictions = model.predict({'video_input': data,
                                     'video_input_masked': data_masked})
    else:
        predictions = model.predict(data_masked)
    predictions = np.ndarray.flatten(predictions)
    stop_time = time.time()
    print(f"Time to get predictions: "
          f"{np.round(stop_time-start_time, 3)} s")

    if len(predictions) != n_frames:
        print(f"predictions len {len(predictions)}, n_frames {n_frames}")

    # now we remove the extra prediction in case the number of frames was not divisible by the window length
    if (n_frames % sliding_window_len) != 0:
        real_predictions = np.zeros(n_frames)
        modulo = n_frames % sliding_window_len
        real_predictions[:len(predictions) - sliding_window_len] = predictions[:len(predictions) - sliding_window_len]
        real_predictions[len(predictions) - sliding_window_len:] = predictions[-modulo:]
        predictions = real_predictions

    return predictions


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


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


def main():
    root_path = None
    with open("param_hne.txt", "r", encoding='UTF-8') as file:
        for nb_line, line in enumerate(file):
            line_list = line.split('=')
            root_path = line_list[1]
    if root_path is None:
        raise Exception("Root path is None")
    path_data = root_path + "data/"
    result_path = root_path + "results_classifier/"
    use_mulimodal_inputs = True
    sliding_window_len = 100

    param = DataForMs(path_data=path_data, result_path=result_path)

    # 3 options to target the cell
    # 1) put the cell in the middle of the frame
    # 2) put all pixels in the border to 1
    # 3) Give 2 inputs, movie full frame (20x20 pixels) + movie mask non binary or binary
    data, data_masked, labels, test_movie_descr = load_data(param=param, sliding_window_len=sliding_window_len,
                                                            with_border=False)
    train_data, valid_data, test_data = data
    train_labels, valid_labels, test_labels = labels
    train_data_masked, valid_data_masked, test_data_masked = data_masked

    train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2],
                                     train_data.shape[3], 1))
    valid_data = valid_data.reshape((valid_data.shape[0], valid_data.shape[1], valid_data.shape[2],
                                     valid_data.shape[3], 1))
    test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], test_data.shape[2],
                                   test_data.shape[3], 1))

    train_data_masked = train_data_masked.reshape((train_data_masked.shape[0], train_data_masked.shape[1],
                                                   train_data_masked.shape[2], train_data_masked.shape[3], 1))
    valid_data_masked = valid_data_masked.reshape((valid_data_masked.shape[0], valid_data_masked.shape[1],
                                                   valid_data_masked.shape[2], valid_data_masked.shape[3], 1))
    test_data_masked = test_data_masked.reshape((test_data_masked.shape[0], test_data_masked.shape[1],
                                                 test_data_masked.shape[2], test_data_masked.shape[3], 1))

    # (sliding_window_size, max_width, max_height, 1)
    # sliding_window in frames, max_width, max_height: in pixel (100, 25, 25, 1) * n_movie
    input_shape = train_data.shape[1:]

    print(f"train_data.shape {train_data.shape}")
    print(f"valid_data.shape {valid_data.shape}")
    # print(f"input_shape {input_shape}")
    print("Data loaded")
    # return
    # building the model
    model = build_model(input_shape, dropout_value=0, use_mulimodal_inputs=use_mulimodal_inputs)
    print(model.summary())

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # print(f"model.layers[0].get_input_shape_at(0) {model.layers[0].get_input_shape_at(0)}")
    # print(f"model.layers[0].get_input_shape_at(1) {model.layers[0].get_input_shape_at(1)}")
    # print(f"model.layers[0].output_shape {model.layers[0].output_shape}")
    # print(f"model.layers[1].output_shape {model.layers[1].output_shape}")
    # print(f"{model.layers[0].output_shape == model.layers[1].output_shape}")
    # return
    n_epochs = 5
    batch_size = 16
    print("Model built and compiled")
    if use_mulimodal_inputs:
        history = model.fit({'video_input': train_data, 'video_input_masked': train_data_masked}, train_labels,
                            epochs=n_epochs,
                            batch_size=batch_size,
                            validation_data=({'video_input': valid_data, 'video_input_masked': valid_data_masked},
                                             valid_labels))
    else:
        history = model.fit(train_data_masked,
                            train_labels,
                            epochs=n_epochs,
                            batch_size=batch_size,
                            validation_data=(valid_data_masked, valid_labels))

    show_plots = True

    if show_plots:
        plot_training_and_validation_loss(history, n_epochs)
        plot_training_and_validation_accuracy(history, n_epochs)

    # used to visualize the results
    if use_mulimodal_inputs:
        prediction = model.predict({'video_input': valid_data,
                                    'video_input_masked': valid_data_masked})
    else:
        prediction = model.predict(valid_data_masked)
    print(f"Validation data: prediction.shape {prediction.shape}")

    for i, valid_label in enumerate(valid_labels):
        print(f"####### video {i}   ##########")
        for j, label in enumerate(valid_label):
            predict_value = str(round(prediction[i, j], 2))
            bonus = ""
            if label == 1:
                bonus = "# "
            print(f"{bonus} f {j}: {predict_value} / {label} ")
        print("")
        print("")

    if use_mulimodal_inputs:
        test_loss, test_acc = model.evaluate({'video_input': test_data,
                                              'video_input_masked': test_data_masked},
                                             test_labels)
    else:
        test_loss, test_acc = model.evaluate(test_data_masked, test_labels)
    print(f"test_acc {test_acc}")

    model.save(f'{param.path_results}transient_classifier_model_acc_test_acc_{test_acc}_{param.time_str}.h5')
    model.save_weights(f'{param.path_results}transient_classifier_weights_acc_test_acc_{test_acc}_{param.time_str}.h5')
    # Save the model architecture
    with open(
            f'{param.path_results}transient_classifier_model_architecture_acc_test_acc_{test_acc}_'
            f'{param.time_str}.json',
            'w') as f:
        f.write(model.to_json())

    if use_mulimodal_inputs:
        prediction = model.predict({'video_input': test_data,
                                    'video_input_masked': test_data_masked})
    else:
        prediction = model.predict(test_data_masked)
    print(f"prediction.shape {prediction.shape}")

    for i, test_label in enumerate(test_labels):
        print(f"####### video {i}  ##########")
        for j, label in enumerate(test_label):
            predict_value = str(round(prediction[i, j], 2))
            bonus = ""
            if label == 1:
                bonus = "# "
            print(f"{bonus} f {j}: {predict_value} / {label} ")
        print("")
        print("")
        # predict_value = str(round(predict_value, 2))
        # print(f"{i}: : {predict_value} / {test_labels[i]}")
    print(f"test_acc {test_acc}")

main()
