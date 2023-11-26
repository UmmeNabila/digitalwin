import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random

# data = pd.read_csv('histories_1.csv')

# def scale_pd(data: pd.DataFrame, qoi: str):
#     min_val = data[qoi].min()
#     max_val = data[qoi].max()
#     return ((data[qoi] - min_val) / (max_val - min_val))

colnames_path = Path.home() /'aims'  / 'digitalwin/colnames.txt'
with colnames_path.open('r') as colnames_file:
    colnames = [colname.strip() for colname in colnames_file.readlines()]


history_path = Path.home() / 'aims' / 'selected_histories'
random_numbers = random.sample(range(1,5001), 50)
random_histories = [f'histories_{num}.csv' for num in random_numbers]

for history in random_histories:
    inpath = Path.home() / 'aims/016_Q8_175_5000_t' / history
    outpath = history_path / history
    
    data = pd.read_csv(inpath)
    selected_data = data[colnames].copy()
    selected_data.to_csv(outpath)
    

# plt.plot(data['time'], scale_pd(data, 'PT1'), label='torque pump 1')
# plt.plot(data['time'], scale_pd(data, 'FL1'), label='flow rate pump 1')
# plt.plot(data['time'], scale_pd(data, 'TA22s1'), label='centerline fuel temp')

# plt.plot(data['time'], scale_pd(data, 'PT2'), label='torque pump 2')
# plt.plot(data['time'], scale_pd(data, 'FL2'), label='flow rate pump 2')
# plt.plot(data['time'], scale_pd(data, 'TA22s1'), label='centerline fuel temp')
# plt.legend()
# plt.show()
