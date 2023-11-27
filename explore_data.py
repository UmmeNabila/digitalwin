import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


# data = pd.read_csv('histories_1.csv')

# def scale_pd(data: pd.DataFrame, qoi: str):
#     min_val = data[qoi].min()
#     max_val = data[qoi].max()
#     return ((data[qoi] - min_val) / (max_val - min_val))

colnames_path = Path.home() /'aims'  / 'digitalwin/colnames.txt'
with colnames_path.open('r') as colnames_file:
    colnames = [colname.strip() for colname in colnames_file.readlines()]


history_path = Path.home() / 'aims' / 'selected_histories'
random.seed(4)
random_numbers = random.sample(range(1,5001), 50)
random_histories = [f'histories_{num}.csv' for num in random_numbers]

# for history in random_histories:
#     inpath = Path.home() / 'aims/016_Q8_175_5000_t' / history
#     outpath = history_path / history
#     data = pd.read_csv(inpath)
#     selected_data = data[colnames].copy()
#     selected_data.to_csv(outpath)

def not_steady_state(sequence: np.array) -> bool:
    mean = sequence.mean()
    percent_difference = (sequence - mean) / mean
    return (percent_difference > .001).any()

def create_sequences(data: np.array, sequence_length: int) -> list:
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        if not_steady_state(sequence):
            sequences.append(sequence)
    return sequences

sequence_length = 100  # This is a hyperparameter, thanks chatgpt <3
sequence_list = []
for history in sorted(history_path.glob('*')):
    centerline_temp = pd.read_csv(history)['TA22s1']
    data = np.array(centerline_temp)
    sequences = create_sequences(data, sequence_length)
    sequence_list.extend(sequences)

random.shuffle(sequence_list)
sequences = np.array(sequence_list)


# split into training and testing (thanks chatgpt)
train_percentage = .8
row_boundary = round(sequences.__len__() * .8)
x_train, y_train = sequences[:row_boundary, :-1], sequences[:row_boundary, -1]
x_test, y_test = sequences[row_boundary:, :-1], sequences[row_boundary:, -1]


# build the RNN model (thanks chatgpt)
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(sequence_length-1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# train the model
model.fit(x_train, y_train, epochs=50, batch_size=32)

# evaluate the model
loss = model.evaluate(x_test, y_test)
print(loss)


# use the model to make predictions

# plt.plot(data['time'], scale_pd(data, 'PT1'), label='torque pump 1')
# plt.plot(data['time'], scale_pd(data, 'FL1'), label='flow rate pump 1')
# plt.plot(data['time'], scale_pd(data, 'TA22s1'), label='centerline fuel temp')

# plt.plot(data['time'], scale_pd(data, 'PT2'), label='torque pump 2')
# plt.plot(data['time'], scale_pd(data, 'FL2'), label='flow rate pump 2')
# plt.plot(data['time'], scale_pd(data, 'TA22s1'), label='centerline fuel temp')
# plt.legend()
# plt.show()
