import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import keras
from string import ascii_lowercase

# fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    inputs = []
    for i in range(window_size, len(series)):
        inputs.append(series[i-window_size:i])
    outputs = series[window_size:]
    return np.asarray(inputs), np.reshape(outputs, (len(outputs),1))

# build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1, activation='linear'))
    return model

# return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?', ' ']
    valid = list(ascii_lowercase) + punctuation
    clean_text = ''
    for char in text:
        if char in valid:
            clean_text += char
        else:
            clean_text += ' '
    return clean_text

### fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    for i in range(window_size, len(text), step_size):
        inputs.append(text[i-window_size:i])
        outputs.append(text[i])
    return inputs, outputs

# build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='linear'))
    model.add(Activation("softmax"))
    return model
