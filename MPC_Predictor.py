from Model_5_18 import LSTM
from Model_width_5_18 import LSTM_Width
import numpy as np
import copy

SAME_STEPS = 20
HORIZON = 10

height_lstm = LSTM()
height_lstm.restore()

width_lstm = LSTM_Width()
width_lstm.restore()

class Predictor():
    def __init__(self, sample_nums, timesteps):
        self.K = sample_nums  # number of sample trajectories
        self.T = timesteps  # timesteps to predict
        self.trajectory_length = 1900
        self.input_sequence_len = height_lstm.TIMESTEPS
        # state
        self.h_state = np.zeros([self.K, 1800, 1])
        self.w_state = np.zeros([self.K, 1800, 1])
        self.Welding_feed_list = np.zeros([self.K, 1900, 1])
        self.Robot_speed_list = np.zeros([self.K, 1900, 1])

        self.count = 0  # reset to zero every catch_up, count how many states have been changed
        self.h_target_list = np.zeros([1800])
        self.w_target_list = np.zeros([1800])

    def get_real_shape(self, Welding_feed_list, Robot_speed_list):
        h_real = height_lstm.welding_pred(Welding_feed_list, Robot_speed_list)
        w_real = width_lstm.welding_pred(Welding_feed_list, Robot_speed_list)
        return h_real, w_real

    def catch_up(self,Welding_feed_list, Robot_speed_list,  h_state, w_state, h_target_list, w_target_list, step):
        """
        update the current state and trajectory history of this episode for this sample agent
        :param h_state: np.array(step + 200, 1)
        :param w_state: np.array(step + 200, 1)
        :param h_target_list: np.array(1700, 1)
        :param w_target_list: np.array(1700, 1)
        :param step: (int) time_step
        :return:
        """
        assert (np.asarray(Welding_feed_list)).shape == (step + 200, )
        assert (np.asarray(Robot_speed_list)).shape == (step + 200, )
        assert (np.asarray(h_state)).shape == (step + 100, )
        assert (np.asarray(w_state)).shape == (step + 100, )
        assert (np.asarray(h_target_list)).shape == (1800, )
        assert (np.asarray(w_target_list)).shape == (1800, )

        # state (for input of the model and for cost)
        self.Welding_feed_list[:, :(step + 200), 0] = Welding_feed_list[:]
        self.Robot_speed_list[:, :(step + 200), 0] = Robot_speed_list[:]
        self.h_state[:, :(step + 100), 0] = h_state[:]
        self.w_state[:, :(step + 100), 0] = w_state[:]

        self.h_target_list = h_target_list
        self.w_target_list = w_target_list

        # how many states it has predicted
        self.count = 0  # reset count

    def cost_fun(self, h_predict_K, w_predict_K, h_target_K, w_target_K):
        # print(np.shape(h_predict_K))
        assert h_predict_K.shape == (self.K, )
        assert w_predict_K.shape == (self.K, )

        cost = 3*np.abs(h_predict_K - h_target_K) + np.abs(w_predict_K - w_target_K)
        return cost

    def predict(self, action, step):
        self.step = step
        assert action.shape == (self.K, 2)
        action_ = copy.copy(action)
        input_welding_feed_list = np.zeros([self.K, self.input_sequence_len, 1])
        input_robot_speed_list = np.zeros([self.K, self.input_sequence_len, 1])

        for k in range(SAME_STEPS):
            self.Welding_feed_list[:, step + self.count + k + 200, 0] \
                = self.Welding_feed_list[:, step + self.count + 199, 0] + action_[:, 0]
            self.Robot_speed_list[:, step + self.count + k + 200, 0] \
                = self.Robot_speed_list[:, step + self.count + 199, 0] + action_[:, 1]

        # update the action data for current state
        # get input sequence for model, the model need self.input_sequence_len steps sequence as input
        # for i in range(self.input_sequence_len):
        input_welding_feed_list[:, :] = self.Welding_feed_list[:, (0 + step + self.count + SAME_STEPS):(self.input_sequence_len + step + self.count + SAME_STEPS)]
        input_robot_speed_list[:, :] = self.Robot_speed_list[:, (0 + step + self.count + SAME_STEPS):(self.input_sequence_len + step + self.count + SAME_STEPS)]

        h_predict_K, w_predict_K = self.Model_predict(input_welding_feed_list, input_robot_speed_list)
        # print(np.shape(w_predict_K))

        # for k in range(10):
        # self.h_state[:, (step + self.count + 100) : (step + self.count + 100 + 10), 0] = copy.copy(h_predict_K[:,0])
        # self.w_state[:, (step + self.count + 100) : (step + self.count + 100 + 10), 0] = copy.copy(w_predict_K[:,0])
        for k in range(SAME_STEPS):
            self.h_state[:, step + self.count + 100 + k, 0] = copy.copy(h_predict_K[:,0])
            self.w_state[:, step + self.count + 100 + k, 0] = copy.copy(w_predict_K[:,0])

        h_target_K = self.h_target_list[step + self.count + 100 + SAME_STEPS - 1]
        w_target_K = self.w_target_list[step + self.count + 100 + SAME_STEPS - 1]
        # compute the cost
        cost = self.cost_fun(h_predict_K[:,0], w_predict_K[:,0], h_target_K, w_target_K)
        assert cost.shape == (self.K, )

        # update count
        self.count += SAME_STEPS
        return cost

    def Model_predict(self, input_welding_feed_list, input_robot_speed_list):
        input_welding_feed_list_s = copy.copy(input_welding_feed_list)
        input_robot_speed_list_s = copy.copy(input_robot_speed_list)
        h_predict_K = height_lstm.welding_pred_batch(np.concatenate((input_welding_feed_list_s, input_robot_speed_list_s), axis = 2))
        w_predict_K = width_lstm.welding_pred_batch(np.concatenate((input_welding_feed_list_s, input_robot_speed_list_s), axis = 2))

        return h_predict_K, w_predict_K



