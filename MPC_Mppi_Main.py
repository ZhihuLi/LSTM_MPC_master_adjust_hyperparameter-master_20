import matplotlib.pyplot as plt
import numpy as np
from MPC_Mppi import MPPI
from sklearn.metrics import mean_squared_error
mppi = MPPI(2048, 200, 0.5)
"""
目标形状设立1700个时间步
"""
STEP_LIMIT = 1500
SAME_STEPS = 20

def target(CHOOSE_NUM):
    h_target_list = []
    w_target_list = []
    CHOOSE_NUM = CHOOSE_NUM

    # 两个都保持不变
    # 共取5条 [1.8,5.3] [1.8,6.3] [2.3,5.3] [2.3,6.3] [2.05, 5.8]
    if CHOOSE_NUM == 1:
        for i in range(1800):
            h_target_list.append(1.8)
            w_target_list.append(5.3)
    if CHOOSE_NUM == 2:
        for i in range(1800):
            h_target_list.append(1.8)
            w_target_list.append(6.3)
    if CHOOSE_NUM == 3:
        for i in range(1800):
            h_target_list.append(2.3)
            w_target_list.append(5.3)
    if CHOOSE_NUM == 4:
        for i in range(1800):
            h_target_list.append(2.3)
            w_target_list.append(6.3)
    if CHOOSE_NUM == 5:
        for i in range(1800):
            h_target_list.append(2.05)
            w_target_list.append(5.8)

    # 一个变，一个不变
    # 共取6条 【高度不变，宽度斜坡】 【高度不变，宽度正弦】 【高度不变，宽度阶跃】， 宽度不变时亦然
    if CHOOSE_NUM == 6:
        for i in range(1800):
            h_target_list.append(2.2)
            w_target_list.append(5.5+i/900)
    if CHOOSE_NUM == 7:
        for i in range(1800):
            h_target_list.append(2.2)
            w_target_list.append(6.5 + np.sin(i/200))
    if CHOOSE_NUM == 8:
        for i in range(1800):
            h_target_list.append(2.2)
            if i < 600:
                w_target_list.append(5.5)
            elif i < 1100:
                w_target_list.append(7.5)
            else:
                w_target_list.append(5.5)
    if CHOOSE_NUM == 9:
        for i in range(1800):
            h_target_list.append(1.7 + (0.7*i)/1800)
            w_target_list.append(5.6)
    if CHOOSE_NUM == 10:
        for i in range(1800):
            h_target_list.append(2.05 + 0.35*np.sin(i/200))
            w_target_list.append(5.6)
    if CHOOSE_NUM == 11:
        for i in range(1800):
            if i < 600:
                h_target_list.append(1.7)
            elif i < 1100:
                h_target_list.append(2.4)
            else:
                h_target_list.append(1.7)
            w_target_list.append(5.6)

    # 两个一起变
    # 共取9条 斜坡，阶跃，正弦两两组合
    if CHOOSE_NUM == 12:
        for i in range(1800):
            h_target_list.append(1.8+(0.5*i)/1800)
            w_target_list.append(5.3+i/1800)
    if CHOOSE_NUM == 13:
        for i in range(1800):
            h_target_list.append(1.8+(0.5*i)/1800)
            w_target_list.append(5.8+0.5*np.sin(i/200))
    if CHOOSE_NUM == 14:
        for i in range(1800):
            h_target_list.append(1.8 + (0.5 * i) / 1800)
            if i < 600:
                w_target_list.append(5.3)
            elif i < 1100:
                w_target_list.append(6.3)
            else:
                w_target_list.append(5.3)
    if CHOOSE_NUM == 15:
        for i in range(1800):
            h_target_list.append(2.05 + 0.25*np.sin(i/200))
            w_target_list.append(5.3+i/1800)
    if CHOOSE_NUM == 16:
        for i in range(1800):
            h_target_list.append(2.05 + 0.25*np.sin(i/200))
            w_target_list.append(5.8+0.5*np.sin(i/200))
    if CHOOSE_NUM == 17:
        for i in range(1800):
            h_target_list.append(2.05 + 0.25 * np.sin(i / 200))
            if i < 600:
                w_target_list.append(5.3)
            elif i < 1100:
                w_target_list.append(6.3)
            else:
                w_target_list.append(5.3)
    if CHOOSE_NUM == 18:
        for i in range(1800):
            if i < 600:
                h_target_list.append(1.8)
            elif i < 1100:
                h_target_list.append(2.3)
            else:
                h_target_list.append(1.8)
            w_target_list.append(5.3+i/1800)
    if CHOOSE_NUM == 19:
        for i in range(1800):
            if i < 600:
                h_target_list.append(1.8)
            elif i < 1100:
                h_target_list.append(2.3)
            else:
                h_target_list.append(1.8)
            w_target_list.append(5.8+0.5*np.sin(i/200))
    if CHOOSE_NUM == 20:
        for i in range(1800):
            if i < 600:
                h_target_list.append(1.8)
            elif i < 1100:
                h_target_list.append(2.3)
            else:
                h_target_list.append(1.8)
            if i < 600:
                w_target_list.append(5.3)
            elif i < 1100:
                w_target_list.append(6.3)
            else:
                w_target_list.append(5.3)
    return h_target_list, w_target_list

def get_real_shape():
    h_real, w_real = mppi.get_real_shape()
    return h_real, w_real

def mppi_main(h_target_list, w_target_list):
    mppi.__init__(2048, 200, 0.5)
    mppi.trajectory_set_initial_value(h_target_list, w_target_list)
    h_target_list_ = h_target_list
    w_target_list_ = w_target_list

    # get shape information from environment
    h_real, w_real = get_real_shape() # 获取目前的形状，焊接开始前将参数设为中位值 [8,12]，以此预测形状
    print("real height and width: ", h_real, w_real)

    mppi.trajectory_set_goal(h_target_list_, w_target_list_)
    mppi.Delta_reset()
    # mppi.trajectory_update_state(h_real, w_real)

    # rollout with mppi algo
    for step in range(STEP_LIMIT):
        if (step % SAME_STEPS == 0):
            print("step: ", step)
            mppi.compute_cost(step)
            target_action_Wf, target_action_Rs = mppi.compute_noise_action()
            mppi.trajectory_update_shape(target_action_Wf, target_action_Rs)
            mppi.Delta_update()
            if step <= 200:
                mppi.Delta_reset()

            mppi.cost_clear()

    h_real_list, w_real_list = mppi.get_real_shape_list()
    Wf_real_list, Rs_real_list = mppi.get_real_parameter_list()
    print("RMSE of height: ", np.sqrt(mean_squared_error(h_target_list[400:1600], h_real_list[400:1600])))
    print("RMSE of width: ", np.sqrt(mean_squared_error(w_target_list[400:1600], w_real_list[400:1600])))

    plt.figure(0)
    plt.plot(h_real_list)
    plt.plot(h_target_list_)
    plt.xlim((0, 2000))
    plt.ylim(0, 5)
    plt.xlabel('time (s)')
    # plt.ylabel('Height(mm)')
    plt.ylabel('Height(mm)')
    plt.title('Control result')

    plt.figure(1)
    plt.plot(w_real_list)
    plt.plot(w_target_list_)
    plt.xlim((0, 2000))
    plt.ylim(0, 10)
    plt.xlabel('time (s)')
    # plt.ylabel('Height(mm)')
    plt.ylabel('Width(mm)')
    plt.title('Control result')
    plt.show()

    plt.figure(2)
    plt.plot(Wf_real_list)
    plt.plot(Rs_real_list)
    plt.xlim((0, 2000))
    plt.ylim(0, 30)
    plt.xlabel('time (s)')
    # plt.ylabel('Height(mm)')
    # plt.ylabel('Width(mm)')
    plt.title('Control result')
    plt.show()

    #
    # plt.figure(3)
    # for i in range(2048):
    #     plt.plot(mppi.NOISE[i,:,0])
    # plt.xlim((0, 2000))
    # plt.ylim(0, 30)
    # plt.xlabel('time (s)')
    # # plt.ylabel('Height(mm)')
    # # plt.ylabel('Width(mm)')
    # plt.title('Control result')
    # plt.show()
    #
    # plt.figure(4)
    # for i in range(2048):
    #     plt.plot(mppi.NOISE[i,:,1])
    # plt.xlim((0, 2000))
    # plt.ylim(0, 30)
    # plt.xlabel('time (s)')
    # # plt.ylabel('Height(mm)')
    # # plt.ylabel('Width(mm)')
    # plt.title('Control result')
    # plt.show()

if __name__ == '__main__':
    h_target_list = []
    w_target_list = []
    # for i in range(1800):
    #     w_target_list.append(5.8+0.5*np.sin(i/200))
    #     h_target_list.append(2.05+0.25*np.sin(i/400))
    #
    # mppi_main(h_target_list, w_target_list)

    for i in range(6, 20):
        h_target_list, w_target_list = target(i+1)
        plt.figure()
        plt.plot(h_target_list)
        plt.title('Target height')
        # plt.show()
        # plt.figure(i)
        plt.plot(w_target_list)
        plt.title('Target width')
        mppi_main(h_target_list, w_target_list)
    plt.show()


