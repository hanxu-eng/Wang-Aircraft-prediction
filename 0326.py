import math

import numpy as np
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.losses import mean_squared_error
from matplotlib import pyplot as plt
from pandas import read_csv
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

time_s = 10  # 拟合后点迹间隔时间
n_data = 8  # 单次输入点迹个数
n_labels = 1  # 单次输出点迹个数
batch_size = 18
epoch = 10

def get_data(n1, n2, time_s, n_data, n_labels):
    seq_data, seq_labels = [], []
    for i in range(n1, n2):
        dataset = read_csv('D:\learn_LSTM\民航航迹100\Variflight_ ({}).csv'.format(i), header=0, index_col=1)
        values = dataset.values
        for j in range(1, 3):
            values = np.delete(values, 1, axis=1)
        values = values.astype('float32')  # 使得所有数值类型都是float类型
        d = np.array(list(values))
        d = d[np.sort(np.unique(d[:, 0], axis=0, return_index=True)[1])]
        Time = d[:, 0]
        height = d[:, 1]
        speed = d[:, 2]
        angle = d[:, 3]
        longitude = d[:, 4]
        latitude = d[:, 5]
        Time = Time - Time[0]

        new_height = interp1d(Time, height, kind='cubic')
        new_speed = interp1d(Time, speed, kind='cubic')
        new_angle = interp1d(Time, angle, kind='cubic')
        new_longitude = interp1d(Time, longitude, kind='cubic')
        new_latitude = interp1d(Time, latitude, kind='cubic')
        new_Time = np.linspace(min(Time), min(Time) + int((max(Time) - min(Time)) / time_s) * time_s,
                               int((max(Time) - min(Time)) / time_s))

        new_height = scaler.fit_transform(np.array(new_height(new_Time)).reshape(-1, 1))
        new_speed = scaler.fit_transform(np.array(new_speed(new_Time)).reshape(-1, 1))
        new_angle = scaler.fit_transform(np.array(new_angle(new_Time)).reshape(-1, 1))
        new_longitude = scaler.fit_transform(np.array(new_longitude(new_Time)).reshape(-1, 1))
        new_latitude = scaler.fit_transform(np.array(new_latitude(new_Time)).reshape(-1, 1))
        new_d = np.stack([new_height, new_speed, new_angle, new_longitude, new_latitude], axis=2)
        new_d = np.array(new_d)
        n_samples = new_d.shape[0] - n_data
        for j in range(n_samples):
            seq_data.append(new_d[j: j + n_data])
            seq_labels.append(new_d[j + n_data: j + n_data + n_labels])
    seq_data = np.array(seq_data).reshape(-1, n_data, 5)
    seq_labels = np.array(seq_labels).reshape(-1, n_labels, 5)
    return seq_data, seq_labels


# 训练数据
train_data, train_labels = get_data(1, 95, time_s, n_data, n_labels)
# 测试数据
test_data, test_labels = get_data(96, 97, time_s, n_data, n_labels)

model = Sequential()
model.add(LSTM(units=50, input_shape=(train_data.shape[1], train_data.shape[2])))
model.add(Dense(5))
# model.add(Activation('ReLU'))
# algorithm = SGD(lr=lr, momentum=momentum)
model.compile(loss='mse', optimizer='SGD')
history = model.fit(train_data, train_labels, epochs=epoch, batch_size=batch_size, verbose=1,
                    shuffle=False)

pre_labels = model(test_data)
test_labels = (test_labels).reshape(-1, 5)
for i in range(5):
    rmse = math.sqrt(mean_squared_error(test_labels[:, i], pre_labels[:, i]))
    print('Test RMSE: %.3f' % rmse)

'''
plt.figure()
m = np.linspace(0, len(test_labels) + 1, len(test_labels))
plt.subplot(231)
plt.plot(m, test_labels[:, 0], label='height', color='r')
plt.plot(m, pre_labels[:, 0], label='pre_height', color='y')
plt.legend()

plt.subplot(232)
plt.plot(m, test_labels[:, 1], label='speed', color='r')
plt.plot(m, pre_labels[:, 1], label='pre_speed', color='y')
plt.legend()

plt.subplot(233)
plt.plot(m, test_labels[:, 2], label='angle', color='r')
plt.plot(m, pre_labels[:, 2], label='pre_angle', color='y')
plt.legend()

plt.subplot(234)
plt.plot(m, test_labels[:, 3], label='longitude', color='r')
plt.plot(m, pre_labels[:, 3], label='pre_longitude', color='y')
plt.legend()

plt.subplot(235)
plt.plot(m, test_labels[:, 4], label='latitude', color='r')
plt.plot(m, pre_labels[:, 4], label='pre_latitudet', color='y')
plt.legend()
plt.show()
'''