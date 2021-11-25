import tensorflow as tf
import numpy as np
import data_api
import base64
import io
import matplotlib.pyplot as plt

from numpy import array
from numpy.random import seed

from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dense

tf.random.set_seed(1)
seed(1)


#
# Convert image to byte string
#
def image_figure(figure):
    pic_io_bytes = io.BytesIO()
    figure.savefig(pic_io_bytes, format='png')
    pic_io_bytes.seek(0)
    res = base64.b64encode(pic_io_bytes.getvalue()).decode("utf-8").replace("\n", "")
    return res


#
# Split data into num_timestamp
#
def data_split(seq, num_timestamp):
    X = []
    Y = []
    for i in range(len(seq)):
        i_end = i + num_timestamp
        if len(seq) - 1 < i_end:
            break
        # i to i_end as input
        # i_end as target output
        x_sequence, y_sequence = seq[i: i_end], seq[i_end]
        X.append(x_sequence)
        Y.append(y_sequence)
    return array(X), array(Y)


def func_factory(model, loss, train_x, train_y, verbose=False):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """

    # Obtain shapes for all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # Prepare required information first for use of tf.dynamic_stitch and tf.dynamic_partition later
    count = 0
    partition = []  # partition indices
    idx = []  # stitch indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
        partition.extend([i] * n)
        count += n

    partition = tf.constant(partition)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, partition, n_tensors)
        for index, (s, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[index].assign(tf.reshape(param, s))

    # Function returned by factory
    @tf.function
    def func(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by func_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # Use GradientTape to calculate the gradient of loss parameters
        with tf.GradientTape() as tape:
            # update parameters in model
            assign_new_model_parameters(params_1d)
            # calculate loss
            loss_value = loss(model(train_x, training=True), train_y)

        # Calculate the gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)

        # Print iteration and loss
        func.iter.assign_add(1)
        if verbose:
            tf.print("Iteration:", func.iter, "loss:", loss_value)

        # Store loss value for later retriever
        tf.py_function(func.history.append, inp=[loss_value], Tout=[])
        return loss_value, grads

    # Information stored as members for use outside this scope
    func.iter = tf.Variable(0)
    func.idx = idx
    func.part = partition
    func.shapes = shapes
    func.assign_new_model_parameters = assign_new_model_parameters
    func.history = []
    return func


#
# Main function
#
def train_data(lat, lon, model_type):
    #
    # Set constants
    #
    num_timestamp = 10
    train_days = 365 * 5  # number of days to train from
    test_days = 365  # number of days to be predicted
    number_of_epochs = 20
    filter_on = 1

    #
    # Plot style
    #
    font = {'family': 'Calibri',
            'weight': 'normal',
            'size': 12
            }
    plt.rc('font', **font)

    data_set = data_api.get_weather_data_at_coordinates(lat, lon)
    col = 'tmax'
    print(data_set[col])
    if filter_on == 1:
        data_set[col] = medfilt(data_set[col], 3)
        data_set[col] = gaussian_filter1d(data_set[col], 1.2)

    #
    # Set number of training and testing data
    #
    train_set = data_set[0: train_days].reset_index(drop=True)
    test_set = data_set[train_days: train_days + test_days].reset_index(drop=True)
    training_set = train_set.iloc[:, 1: 2].values
    testing_set = test_set.iloc[:, 1: 2].values

    #
    # Normalize data first
    #
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    testing_set_scaled = sc.fit_transform(testing_set)

    x_train, y_train = data_split(training_set_scaled, num_timestamp)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test, y_test = data_split(testing_set_scaled, num_timestamp)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Single cell LSTM
    if model_type == 1:
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(x_train.shape[1], 1)))
        model.add(Dense(units=1))

    # Stacked LSTM
    elif model_type == 2:
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))

    # Bidirectional LSTM
    elif model_type == 3:
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(x_train.shape[1], 1)))
        model.add(Dense(1))
    else:
        raise Exception('model type is not supported')

    #
    # Begin training
    #
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(x_train, y_train, epochs=number_of_epochs, batch_size=32)
    loss = history.history['loss']
    epochs = range(len(loss))

    #
    # Get predicted data
    #
    y_predicted = model.predict(x_test)

    #
    # Reverse normalize the data
    #
    y_predicted_descaled = sc.inverse_transform(y_predicted)

    # y_train_descaled = sc.inverse_transform(y_train)
    y_test_descaled = sc.inverse_transform(y_test)
    # y_pred = y_predicted.ravel()
    # y_pred = [round(yx, 2) for yx in y_pred]
    # y_tested = y_test.ravel()

    #
    # Create result figure and return it
    #
    plt.figure(figsize=(16, 14))

    plt.subplot(3, 1, 1)
    plt.plot(data_set[col], color='blue', linewidth=1, label='Actual Value')
    plt.title("All data")
    plt.xlabel("Day")
    plt.ylabel("Temperature")

    plt.subplot(3, 2, 3)
    plt.plot(y_test_descaled, color='blue', linewidth=1, label='Actual Value')
    plt.plot(y_predicted_descaled, color='darkorange', linewidth=1, label='Predicted Value')
    plt.legend(frameon=False)
    plt.title("Test data (365 days)")
    plt.xlabel("Day")
    plt.ylabel("Temperature")

    plt.subplot(3, 2, 4)
    plt.plot(y_test_descaled[0:100], color='blue', linewidth=1, label='Actual Value')
    plt.plot(y_predicted_descaled[0:100], color='darkorange', label='Predicted Value')
    plt.legend(frameon=False)
    plt.title("Test data (first 100 days)")
    plt.xlabel("Day")
    plt.ylabel("Temperature")

    plt.subplot(3, 3, 7)
    plt.plot(epochs, loss, color='teal')
    plt.title("Training curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")

    plt.subplot(3, 3, 8)
    plt.plot(y_test_descaled - y_predicted_descaled, color='red')
    plt.title("Delta")
    plt.xlabel("Day")
    plt.ylabel("Delta")

    plt.subplot(3, 3, 9)
    plt.scatter(y_predicted_descaled, y_test_descaled, s=2, color='indigo')
    plt.title("Scatter plot")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    image = image_figure(plt)
    plt.close()
    mse = mean_squared_error(y_test_descaled, y_predicted_descaled)
    r2 = r2_score(y_test_descaled, y_predicted_descaled)
    print("mse=" + str(round(mse, 2)))
    print("r2=" + str(round(r2, 2)))
    return f"data:image/png;base64,{image}", list(y_predicted_descaled)[-1][0]
