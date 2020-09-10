import numpy as np
import copy
from Model_5_18 import LSTM
from Model_width_5_18 import LSTM_Width

SAME_STEPS = 20

height_lstm = LSTM()
height_lstm.restore()

width_lstm = LSTM_Width()
width_lstm.restore()

class Trajectory():
    """
    collect trajectory history and preprocess the data making it more suitable for the input of predictor
    """
    def __init__(self):

        self.h_target_list = np.zeros([1800])
        self.w_target_list = np.zeros([1800])
        self.count = 0

    def set_initial_value(self, h_target_list, w_target_list):
        raw_data = np.loadtxt('initial_value.txt')
        raw_data = np.array(raw_data)
        Error_list = np.abs(raw_data[:, 2] - h_target_list[99]) + np.abs(raw_data[:, 3] - w_target_list[99])
        self.Welding_feed_list = [raw_data[np.argmin(Error_list), 0]] * 200
        self.Robot_speed_list = [raw_data[np.argmin(Error_list), 1]] * 200
        self.h_state = [raw_data[np.argmin(Error_list), 2]]*100
        self.w_state = [raw_data[np.argmin(Error_list), 3]]*100

    def set_goal(self, h_target_list, w_target_list):
        self.h_target_list = h_target_list
        self.w_target_list = w_target_list

    def update_state(self, h_real, w_real):
        h_real = copy.copy(h_real)
        w_real = copy.copy(w_real)

        if self.count > 0:
            self.h_state.append(h_real)
            self.w_state.append(w_real)
            self.count += 1
        # update state

    def get_h_state(self):
        h_state = copy.copy(self.h_state)
        return np.asarray(h_state)

    def get_w_state(self):
        w_state = copy.copy(self.w_state)
        return np.asarray(w_state)

    def get_h_target(self):
        return self.h_target_list

    def get_w_target(self):
        return self.w_target_list

    def update_welding_parameter(self, target_action_Wf, target_action_Rs):
        target_action_Wf_ = copy.copy(target_action_Wf)
        target_action_Rs_ = copy.copy(target_action_Rs)
        Welding_feed_list_batch = []
        Robot_speed_list_batch = []

        for k in range(SAME_STEPS):
            self.Welding_feed_list.append(self.Welding_feed_list[-SAME_STEPS] + target_action_Wf_)
            self.Robot_speed_list.append(self.Robot_speed_list[-SAME_STEPS] + target_action_Rs_)
            Welding_feed_list_batch.append(self.Welding_feed_list[-200:])
            Robot_speed_list_batch.append(self.Robot_speed_list[-200:])
        Welding_feed_list_batch = np.reshape(Welding_feed_list_batch,(SAME_STEPS,200,1))
        Robot_speed_list_batch = np.reshape(Robot_speed_list_batch,(SAME_STEPS,200,1))
        h_real_batch = height_lstm.welding_pred_batch(np.concatenate((Welding_feed_list_batch, Robot_speed_list_batch), axis = 2))
        w_real_batch = width_lstm.welding_pred_batch(np.concatenate((Welding_feed_list_batch, Robot_speed_list_batch), axis = 2))
        for k in range(SAME_STEPS):
            self.h_state.append(h_real_batch[k])
            self.w_state.append(w_real_batch[k])




