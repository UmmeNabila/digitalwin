import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('histories_1.csv')


def scale_pd(data: pd.DataFrame, qoi: str):
    min_val = data[qoi].min()
    max_val = data[qoi].max()
    return ((data[qoi] - min_val) / (max_val - min_val))


plt.plot(data['time'], scale_pd(data, 'PT1'), label='torque pump 1')
plt.plot(data['time'], scale_pd(data, 'FL1'), label='flow rate pump 1')
plt.plot(data['time'], scale_pd(data, 'TA22s1'), label='centerline fuel temp')

# plt.plot(data['time'], scale_pd(data, 'PT2'), label='torque pump 2')
# plt.plot(data['time'], scale_pd(data, 'FL2'), label='flow rate pump 2')
# plt.plot(data['time'], scale_pd(data, 'TA22s1'), label='centerline fuel temp')
plt.legend()
plt.show()
