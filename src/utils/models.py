import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tabulate import tabulate


class CLRScheduler(tf.keras.callbacks.Callback):
  """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

  def __init__(self, schedule):
    super(CLRScheduler, self).__init__()
    self.schedule = schedule

  def on_epoch_begin(self, epoch, logs=None):
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    # Get the current learning rate from model's optimizer.
    lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
    # Call schedule function to get the scheduled learning rate.
    scheduled_lr = self.schedule(epoch, lr)
    # Set the value back to the optimizer before this epoch starts
    tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
    print('\nEpoch %05d: Learning rate is %6.4f.' % (epoch, scheduled_lr))


def plot_metric_function(history, metric, name):
    """Plots a models metric functions"""
    figure = plt.figure(figsize=(8, 8))
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title(name + ' ' + metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def top_1_table(model, model_name, history, dataset_name):
    table = [[dataset_name, model_name, max(history.history['acc']), max(history.history['loss']), 'TBD']]
    print(tabulate(table, headers=['Dataset', 'Model', 'Top-1 acc.', 'Top-1 loss', 'Model Size (GB)']))


def triangular_clr(epoch, lr):
    """triangular cyclical learning rate, to be used in learningratescheduler callback
    note that min_lr, max_lr, step_size and model are hardcoded: replace"""
    min_lr = 1e-7 # replace by desired lr
    max_lr = 1e-2 # replace by desired lr
    step_size = 4 # replace by desired lr

    diff = max_lr - min_lr

    increase = (diff / step_size) * 2

    if epoch == 0:
        lr = min_lr
    elif ((epoch % step_size) / step_size) < 0.5:
        lr += increase
    else:
        lr -= increase
    
    print(epoch, lr)

    return lr

def run(model, path, train_data, train_labels, val_data, val_labels, name, learning_rate):
    """Compiles, trains, and saves ML model.
    In addition, plots loss and accuracy."""

    ## Create optimizer
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)

    ## Compile model
    model.compile(optimizer=adam,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    ## Summarization
    model.summary()

    ## Training
    history = model.fit(
        train_data,
        train_labels,
        epochs=25, # used to be 25, set to 1 for testing purposes
        validation_data=(val_data, val_labels),
        batch_size=64, # used to be 64, tried to decrease memory allocation size
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3),
            CLRScheduler(triangular_clr)
            ]
        )

    ## Save weights to binary file
    model.save_weights(filepath = path + name + '_weights.h5')

    # serialize model to JSON
    model_json = model.to_json()
    with open(path + name + '.json', 'w') as json_file:
        json_file.write(model_json)

    ## plot loss and accuracy
    plot_metric_function(history, 'loss', name)
    plot_metric_function(history, 'acc', name)

    ## print top-1 accuracy and loss table
    top_1_table(model, name, history, ("Stanford40" if name == "image_model" else "TV-HI"))

def create_transfer_model(trained_model, path, n_classes):
    """Creates a transfer-learning model based on trained_model 
    and makes it ready for fine-tuning"""

    # load pre-trained model from json
    json_file = open(path + trained_model + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(path + trained_model + '_weights.h5')

    # create new model based on trained_model
    X_input = loaded_model.input
    X = loaded_model.layers[-3].output
    X = keras.layers.Dense(n_classes, name="dense_2")(X)
    X = keras.layers.Activation('softmax')(X)
    model = keras.models.Model(inputs=X_input, outputs=X)

    for layer in model.layers[:-2]:
        layer.trainable=False
    
    return model

def create_two_stream_model(stream_1, stream_2, path, n_classes):
    """Creates a two-stream model 'model' of streams stream_1 and
    stream_2, concatenated at output layers, with output classes n_classes"""

    # load stream 1 from json
    json_file_1 = open(path + stream_1 + '.json', 'r')
    loaded_model_1_json = json_file_1.read()
    json_file_1.close()

    loaded_model_1 = keras.models.model_from_json(loaded_model_1_json)
    loaded_model_1.load_weights(path + stream_1 + '_weights.h5')

    # load stream 2 from json
    json_file_2 = open(path + stream_2 + '.json', 'r')
    loaded_model_2_json = json_file_2.read()
    json_file_2.close()

    loaded_model_2 = keras.models.model_from_json(loaded_model_2_json)
    loaded_model_2.load_weights(path + stream_2 + '_weights.h5')

    # concatenate the two streams at the last convolutional layer
    X_input = [loaded_model_1.input, loaded_model_2.input]
    X = keras.layers.concatenate([loaded_model_1.layers[-4].output, loaded_model_2.layers[-4].output])
    X = keras.layers.Conv2D(64, kernel_size=5, activation="tanh", kernel_regularizer=keras.regularizers.l2(0.001))(X)
    X = keras.layers.MaxPool2D(pool_size=2)(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(32)(X)
    X = keras.layers.Dense(16)(X)
    X = keras.layers.Dense(n_classes)(X)
    X = keras.layers.Activation('softmax')(X)
    model = keras.models.Model(inputs=X_input, outputs=X)

    for layer in model.layers[:-7]:
        layer.trainable=False

    return model
