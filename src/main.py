import os

import numpy as np

import tensorflow as tf
from tensorflow import keras

from utils.preprocessing import import_images, import_videos, split_validation, split_validation_video
from utils.models import plot_metric_function, run, create_transfer_model,create_two_stream_model


# 1. GLOBAL VARIABLES

# paths
DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep

SAVED_DATA = DATA_PATH + "saved" + os.sep

DATA_IMAGES = DATA_PATH + "JPEGImages" + os.sep
LIST_IMAGES_T = DATA_PATH + "ImageSplits" + os.sep + "train.txt"
LIST_IMAGES_V = DATA_PATH + "ImageSplits" + os.sep + "test.txt"

DATA_VIDEOS = DATA_PATH + "tv_human_interactions_videos" + os.sep
LIST_VIDEOS = DATA_PATH + "readme.txt"

MODEL_PATH = os.path.join(os.getcwd(), 'models') + os.sep

# dimensions of resized images and frames
DIM = (224,224)

# learning rates
LR_1 = 0.0001
LR_2 = LR_1 * 0.1


# 2. DATASET: IMAGES

# Import train & validation set
# check if data isn't already saved
if (
    not (
        os.path.isfile(SAVED_DATA + 'image_classes.npy') and
        os.path.isfile(SAVED_DATA + 'train_images.npy') and
        os.path.isfile(SAVED_DATA + 'val_images.npy') and
        os.path.isfile(SAVED_DATA + 'train_image_labels.npy') and
         os.path.isfile(SAVED_DATA + 'val_image_labels.npy')       
        )
    ):
    print('Importing image train and val sets . . .')

    images, labels, image_classes = import_images(LIST_IMAGES_T, DATA_IMAGES, DIM)
    train_images, val_images, train_image_labels, val_image_labels = split_validation(images, labels, 0.1)

    np.save(SAVED_DATA + 'image_classes.npy', image_classes)
    np.save(SAVED_DATA + 'train_images.npy', train_images)
    np.save(SAVED_DATA + 'val_images.npy', val_images)
    np.save(SAVED_DATA + 'train_image_labels.npy', train_image_labels)
    np.save(SAVED_DATA + 'val_image_labels.npy', val_image_labels)

    del images, labels # not necessary anymore

    print('Image train and val sets successfully imported.')
else:
    print('Loading existing image train and val sets . . .')

    image_classes = np.load(SAVED_DATA + 'image_classes.npy')
    train_images = np.load(SAVED_DATA + 'train_images.npy')
    val_images = np.load(SAVED_DATA + 'val_images.npy')
    train_image_labels = np.load(SAVED_DATA + 'train_image_labels.npy')
    val_image_labels = np.load(SAVED_DATA + 'val_image_labels.npy')

    print('Existing image train and val sets successfully loaded.')

# Import test set
# check if data isn't already saved
if (
    not (
        os.path.isfile(SAVED_DATA + 'test_images.npy') and
        os.path.isfile(SAVED_DATA + 'test_labels.npy')   
        )
    ):
    print('Importing image test set . . .')

    test_images, test_labels, test_classes = import_images(LIST_IMAGES_V, DATA_IMAGES, DIM)

    np.save(SAVED_DATA + 'test_images.npy', test_images)
    np.save(SAVED_DATA + 'test_labels.npy', test_labels)

    del test_classes # is the same as image_classes

    print('Image test set successfully imported.')
else:
    print('Loading existing image test set . . .')

    test_images = np.load(SAVED_DATA + 'test_images.npy')
    test_labels = np.load(SAVED_DATA + 'test_labels.npy')

    print('Existing image test set successfully loaded.')


# 3. DATASET: VIDEOS

# Import train & validation set
# check if data isn't already saved
if (
    not (
        os.path.isfile(SAVED_DATA + 'video_classes.npy') and
        os.path.isfile(SAVED_DATA + 'train_m_frames.npy') and
        os.path.isfile(SAVED_DATA + 'val_m_frames.npy') and
        os.path.isfile(SAVED_DATA + 'train_video_labels.npy') and
        os.path.isfile(SAVED_DATA + 'val_video_labels.npy') and
         os.path.isfile(SAVED_DATA + 'train_flow.npy') and
        os.path.isfile(SAVED_DATA + 'val_flow.npy') and
        os.path.isfile(SAVED_DATA + 'train_flow_labels.npy') and
        os.path.isfile(SAVED_DATA + 'val_flow_labels.npy')
        )
    ):
    print('Importing video train and val sets . . .')

    m_frames, optical_flow, labels, video_classes = import_videos(LIST_VIDEOS, DATA_VIDEOS, DIM, 24, 28, 16)
    train_m_frames, val_m_frames, train_video_labels, val_video_labels, train_flow, val_flow, train_flow_labels, val_flow_labels = split_validation_video(
        m_frames,
        labels,
        optical_flow,
        labels,
        0.1
        )

    np.save(SAVED_DATA + 'video_classes.npy', video_classes)
    np.save(SAVED_DATA + 'train_m_frames.npy', train_m_frames)
    np.save(SAVED_DATA + 'val_m_frames.npy', val_m_frames)
    np.save(SAVED_DATA + 'train_video_labels.npy', train_video_labels)
    np.save(SAVED_DATA + 'val_video_labels.npy', val_video_labels)
    np.save(SAVED_DATA + 'train_flow.npy', train_flow)
    np.save(SAVED_DATA + 'val_flow.npy', val_flow)
    np.save(SAVED_DATA + 'train_flow_labels.npy', train_flow_labels)
    np.save(SAVED_DATA + 'val_flow_labels.npy', val_flow_labels)

    del m_frames, optical_flow, labels # redundant

    print('Video train and val sets successfully imported.')
else:
    print('Loading existing video train and val sets . . .')

    video_classes = np.load(SAVED_DATA + 'video_classes.npy')
    train_m_frames = np.load(SAVED_DATA + 'train_m_frames.npy')
    val_m_frames = np.load(SAVED_DATA + 'val_m_frames.npy')
    train_video_labels = np.load(SAVED_DATA + 'train_video_labels.npy')
    val_video_labels = np.load(SAVED_DATA + 'val_video_labels.npy')
    train_flow = np.load(SAVED_DATA + 'train_flow.npy')
    val_flow = np.load(SAVED_DATA + 'val_flow.npy')
    train_flow_labels = np.load(SAVED_DATA + 'train_flow_labels.npy')
    val_flow_labels = np.load(SAVED_DATA + 'val_flow_labels.npy')

    print('Existing video train and val sets successfully loaded.')


# Import test set
# check if data isn't already saved
if (
    not (
        os.path.isfile(SAVED_DATA + 'test_m_frames.npy') and
        os.path.isfile(SAVED_DATA + 'test_flow.npy') and
        os.path.isfile(SAVED_DATA + 'test_video_labels.npy')   
        )
    ):
    print('Importing video test set . . .')

    test_m_frames, test_flow, test_video_labels, test_classes = import_videos(LIST_VIDEOS, DATA_VIDEOS, DIM, 17, 21, 16)

    np.save(SAVED_DATA + 'test_m_frames.npy', test_m_frames)
    np.save(SAVED_DATA + 'test_flow.npy', test_flow)
    np.save(SAVED_DATA + 'test_video_labels.npy', test_video_labels)

    del test_classes # is the same as video_classes

    print('Video test set successfully imported.')
else:
    print('Loading existing video test set . . .')

    test_m_frames = np.load(SAVED_DATA + 'test_m_frames.npy')
    test_flow = np.load(SAVED_DATA + 'test_flow.npy')
    test_video_labels = np.load(SAVED_DATA + 'test_video_labels.npy')

    print('Existing video test set successfully loaded.')


# 4. CNN: IMAGES

# check if model isn't already trained and saved
if not (os.path.isfile(MODEL_PATH + 'image_model.json') and os.path.isfile(MODEL_PATH + 'image_model_weights.h5')):
    
    print('Training image model . . .')
    # calculate number of output classes
    n_classes = image_classes.shape[0]

    # create new model
    X_input = keras.layers.Input((224,224,3), name='video_input') # names provided for two-stream model later
    X = X_input
    X = keras.layers.Conv2D(16, kernel_size=5, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001), name='video_conv2d_1')(X)
    X = keras.layers.MaxPooling2D(pool_size=(2), name='video_pool_1')(X)
    X = keras.layers.Conv2D(32, kernel_size=5, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001), name='video_conv2d_2')(X)
    X = keras.layers.MaxPooling2D(pool_size=(2), name='video_pool_2')(X)
    X = keras.layers.Conv2D(64, kernel_size=5, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001), name='video_conv2d_3')(X)
    X = keras.layers.Conv2D(64, kernel_size=5, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001), name='video_conv2d_4')(X)
    X = keras.layers.Flatten(name='video_flatten_1')(X)
    X = keras.layers.Dense(n_classes)(X)    
    X = keras.layers.Activation("softmax")(X)
    image_model = keras.models.Model(inputs=X_input,outputs=X)

    # run model
    run(image_model, MODEL_PATH, train_images, train_image_labels, val_images, val_image_labels, 'image_model', LR_1)
    print('Image model successfully trained.')

else:
    print('Using existing image model.')


# 5. CNN: VIDEOS

# check if model isn't already trained and saved
if not (os.path.isfile(MODEL_PATH + 'video_model.json') and os.path.isfile(MODEL_PATH + 'video_model_weights.h5')):
    # check if image model is available
    if os.path.isfile(MODEL_PATH + 'image_model.json') and os.path.isfile(MODEL_PATH + 'image_model_weights.h5'):
        
        print('Fine-tuning video model . . .')
        # calculate number of output classes
        n_classes = video_classes.shape[0]

        # create new model
        video_model = create_transfer_model('image_model', MODEL_PATH, n_classes)
        # video_model = keras.Sequential()

        # # create transfer learning model
        # create_transfer_model(video_model, 'image_model', MODEL_PATH, n_classes)

        # tune model
        run(video_model, MODEL_PATH, train_m_frames, train_video_labels, val_m_frames, val_video_labels, 'video_model', LR_2)
        print('Video model successfully tuned.')
    
    else:
        print('ERROR: Image model is not available for transfer learning.')

else:
    print('Using existing video model.')


# 6. CNN: OPTICAL FLOW

# check if model isn't already trained and saved
if not (os.path.isfile(MODEL_PATH + 'flow_model.json') and os.path.isfile(MODEL_PATH + 'flow_model_weights.h5')):
    
    print('Training optical flow model . . .')
    # calculate number of output classes
    n_classes = video_classes.shape[0]

    # create new model
    X_input = keras.layers.Input((224,224,16))
    X = X_input
    X = keras.layers.Conv2D(16, kernel_size=5, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001), name='flow_conv2d_1')(X)
    X = keras.layers.MaxPooling2D(pool_size=(2), name='flow_pool_1')(X)
    X = keras.layers.Conv2D(32, kernel_size=5, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001), name='flow_conv2d_2')(X)
    X = keras.layers.MaxPooling2D(pool_size=(2), name='flow_pool_2')(X)
    X = keras.layers.Conv2D(64, kernel_size=5, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001), name='flow_conv2d_3')(X)
    X = keras.layers.Conv2D(64, kernel_size=5, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001), name='flow_conv2d_4')(X)
    X = keras.layers.Flatten(name='flow_flatten_1')(X)
    X = keras.layers.Dense(n_classes)(X)    
    X = keras.layers.Activation("softmax")(X)
    flow_model = keras.models.Model(inputs=X_input,outputs=X)

    # run model
    run(flow_model, MODEL_PATH, train_flow, train_flow_labels, val_flow, val_flow_labels, 'flow_model', LR_1)
    print('Optical flow model successfully trained.')

else:
    print('Using existing optical flow model.')


# 7. CNN: TWO-STREAM

# check if model isn't already trained and saved
if not (os.path.isfile(MODEL_PATH + 'two_stream_model.json') and os.path.isfile(MODEL_PATH + 'two_stream_model_weights.h5')):
    # check if video model and optical flow model are available
    if (
        os.path.isfile(MODEL_PATH + 'video_model.json') and
        os.path.isfile(MODEL_PATH + 'video_model_weights.h5') and
        os.path.isfile(MODEL_PATH + 'flow_model.json') and
        os.path.isfile(MODEL_PATH + 'flow_model_weights.h5')
    ):
        print('Training two-stream model . . .')
        # calculate number of output classes
        n_classes = video_classes.shape[0]

        # create new model
        two_stream_model = create_two_stream_model('video_model', 'flow_model', MODEL_PATH, n_classes)

        # tune model
        run(
            two_stream_model,
            MODEL_PATH,
            [train_m_frames, train_flow],
            train_video_labels,
            [val_m_frames, val_flow],
            val_video_labels,
            'two_stream_model',
            LR_2
            )
        print('Two-stream model successfully trained.')

        pred = two_stream_model.evaluate([test_m_frames, test_flow], test_video_labels)
        print(pred)
        print(two_stream_model.metrics_names)


    
    else:
        print('ERROR: Video and/or flow model is/are not available for two-stream connection.')

else:
    print('Using existing two-stream model.')
