import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping


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

def mask_steady_state(sequences: np.array) -> np.array:
    means = sequences.mean(1)
    means = np.expand_dims(means, axis=1)
    percent_difference = (sequences - means) / means
    return (percent_difference > .001).any(1)

def create_sequences(data: np.array, sequence_length: int) -> list:
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        # if not_steady_state(sequence):
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


# split into training and testing
# ts stands for transient state (steady state sequences removed)
train_percentage = .75
train_boundary = round(sequences.__len__() * train_percentage)
validation_percentage = .05
val_boundary = round(sequences.__len__() * (1-validation_percentage))

train = sequences[:train_boundary, :]
train_ts = train[mask_steady_state(train), :]
x_train, y_train = train[:, :-1], train[:, -1]
x_train_ts, y_train_ts = train_ts[:, :-1], train_ts[:, -1]

test = sequences[train_boundary:val_boundary, :]
test_ts = test[mask_steady_state(test), :]
x_test, y_test = test[:, :-1], test[:, -1]
x_test_ts, y_test_ts = test_ts[:, :-1], test_ts[:, -1]

val = sequences[val_boundary:, :]
x_val, y_val = val[:, :-1], val[:, -1]


# build the RNN models (thanks chatgpt)
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(sequence_length-1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model_ts = Sequential()
model_ts.add(SimpleRNN(50, activation='relu', input_shape=(sequence_length-1, 1)))
model_ts.add(Dense(1))
model_ts.compile(optimizer='adam', loss='mse')


# setup callbacks: checkpoints and early stopping
checkpoint = ModelCheckpoint(
    'rnn_checkpoint.h5',        # Path where to save the model
    monitor='val_loss',         # Metric to monitor
    verbose=1,                  # Verbosity mode
    save_best_only=True,        # Save only the best model
    save_weights_only=True,     # If True, only weights are saved
    mode='auto',                # Auto mode means the direction is inferred
    period=1                    # Checkpoint interval (every epoch in this case)
)

checkpoint_ts = ModelCheckpoint(
    'rnn_ts_checkpoint.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    period=1
)

early_stopping = EarlyStopping(
    monitor='val_loss', 
    min_delta=0.001, 
    patience=20,
    verbose=1,
    mode='auto'
)


# train the models
model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs = 100,
    batch_size=50,
    callbacks=[checkpoint, early_stopping],
)

model_ts.fit(
    x_train_ts, y_train_ts,
    validation_data=(x_val, y_val),
    epochs = 100,
    batch_size=50,
    callbacks=[checkpoint_ts, early_stopping],
)

# evaluate the models
ss_ss_loss = model.evaluate(x_test, y_test)
ss_ts_loss = model.evaluate(x_test_ts, y_test_ts)
ts_ss_loss = model_ts.evaluate(x_test, y_test)
ts_ts_loss = model_ts.evaluate(x_test_ts, y_test_ts)
total_sequences = sequences.__len__()
ts_sequences = sequences[mask_steady_state(sequences)].__len__()

print(f'total sequences: {total_sequences}')
print(f'percentage transient: {ts_sequences/total_sequences}')
print(f'full data, full test loss: {ss_ss_loss}')
print(f'full data, transient test loss: {ss_ts_loss}')
print(f'transient data, full test loss: {ts_ss_loss}')
print(f'transient data, transient test loss: {ts_ts_loss}')

# plt.plot(data['time'], scale_pd(data, 'PT1'), label='torque pump 1')
# plt.plot(data['time'], scale_pd(data, 'FL1'), label='flow rate pump 1')
# plt.plot(data['time'], scale_pd(data, 'TA22s1'), label='centerline fuel temp')

# plt.plot(data['time'], scale_pd(data, 'PT2'), label='torque pump 2')
# plt.plot(data['time'], scale_pd(data, 'FL2'), label='flow rate pump 2')
# plt.plot(data['time'], scale_pd(data, 'TA22s1'), label='centerline fuel temp')
# plt.legend()
# plt.show()
