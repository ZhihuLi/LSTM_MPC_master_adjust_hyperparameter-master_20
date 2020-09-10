from Model_5_18 import LSTM
from Model_width_5_18 import LSTM_Width
import numpy as np
import matplotlib.pyplot as plt

height_lstm = LSTM()
height_lstm.restore()

width_lstm = LSTM_Width()
width_lstm.restore()

def new_data_for_net():
    # 送丝速度的变化范围3 - 13，共有101个值，焊接速度的变化范围3-26, 共有231个值
    # 比例的变化范围为1-2
    # number = 0
    input_net = np.zeros((23331, 2), float)
    for i in range(23331):
        input_net[i][0] = 3 + (int(i / 231)) * 0.1
        input_net[i][1] = 3 + (i % 231) * 0.1
        if input_net[i][1]/ input_net[i][0] > 2 or input_net[i][1]/ input_net[i][0] < 1 :
            input_net[i][0] = 0
            input_net[i][1] = 0
        input_net_not0 = []
    for i in range(len(input_net)):
        if (input_net[i,0]>0) &(input_net[i, 1] > 0):
            input_net_not0.append(input_net[i])
    np.savetxt('linear_input_net.txt',input_net_not0, fmt='%.1f %.1f', delimiter='\n')
    return input_net_not0

if __name__ == '__main__':
    input_net = new_data_for_net()
    print(np.shape(input_net))
    input_net = np.array(input_net)
    input_net_200 = np.zeros((8166, 200, 2))
    for i in range(200):
        input_net_200[:,i,:] = input_net[:,:]
    h_predict_all = height_lstm.welding_pred_batch(input_net_200)
    w_predict_all = width_lstm.welding_pred_batch(input_net_200)
    print(h_predict_all)
    all_data = np.zeros((len(h_predict_all), 4))
    all_data[:, 0:2] = input_net
    for i in range(0, len(h_predict_all)):
        all_data[i, 2] = h_predict_all[i]
        all_data[i, 3] = w_predict_all[i]
    np.savetxt('initial_value.txt', all_data, fmt='%.1f %.1f %.3f %.3f', delimiter='\n')

    plt.figure(0)
    plt.scatter(h_predict_all[:], w_predict_all[:])
    # plt.xlim((0, 2000))
    # plt.ylim(0, 10)
    plt.xlabel('time (s)')
    # plt.ylabel('Height(mm)')
    plt.ylabel('Width(mm)')
    plt.title('Control result')
    plt.show()