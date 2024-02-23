from scipy.optimize import minimize
import numpy as np
import pandas as pd
import random
import time
from scipy import optimize as op
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import math




results = []
A1 = np.array([[8.0,10.9],
[9.9,9.5]])

D1 = np.array([[-26.5,-32.1],
[-37.4,-31.7]])
A2 = np.array([[8.6,12.9],
[10.5,6.5]])

D2 = np.array([[-24.6,-36.1],
[-35.3,-24.9]])
ALPHA = 0.5
p = 0.7
EPSILON = 0.7
res_process = [ 0, 0, 0, 0, 0, 1, 0.5, 0.5]
# global variables
N_STATES = 1  # the length of the 1 dimentional world
ACTIONS_S1 = ['a1,d1', 'a1,d2', 'a2,d1', 'a2,d2']  # available actions

def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions)
   # print(table)
    return table
def build_q_table_d(n_states, actions):
    table = pd.DataFrame(
        -50*np.ones((n_states, len(actions))),
        columns=actions)
   # print(table)
    return table
def get_env_feedback1_a_1(action_S1):
    # This is how agent will interact with the environment
    R_a1 = 0
    R_d1 = 0
    if action_S1 == 'a1,d1':
        R_a1 = A1[0][0]
        R_d1 = D1[0][0]
    if action_S1 == 'a1,d2':
        R_a1 = A1[0][1]
        R_d1 = D1[0][1]
    if action_S1 == 'a2,d1':
        R_a1 = A1[1][0]
        R_d1 = D1[1][0]
    if action_S1 == 'a2,d2':
        R_a1 = A1[1][1]
        R_d1 = D1[1][1]
    return R_a1, R_d1
def get_env_feedback1_a_2(action_S1):
    # This is how agent will interact with the environment
    R_a2 = 0
    R_d2 = 0
    if action_S1 == 'a1,d1':
        R_a2 = A2[0][0]
        R_d2 = D2[0][0]
    if action_S1 == 'a1,d2':
        R_a2 = A2[0][1]
        R_d2 = D2[0][1]
    if action_S1 == 'a2,d1':
        R_a2 = A2[1][0]
        R_d2 = D2[1][0]
    if action_S1 == 'a2,d2':
        R_a2 = A2[1][1]
        R_d2 = D2[1][1]
    return R_a2, R_d2

def objective(x):
    y1 = (p*q_table_S1_a_1.loc[0, 'a1,d1']+(1-p)*q_table_S1_a_2.loc[0, 'a1,d1']) * x[2] * x[6] + (p*q_table_S1_a_1.loc[0, 'a1,d1']+(1-p)*q_table_S1_a_2.loc[0, 'a1,d2']) * x[3] * x[6] + (p*q_table_S1_a_1.loc[0, 'a1,d2']+(1-p)*q_table_S1_a_2.loc[0, 'a1,d1']) * x[4] * x[6] + (p*q_table_S1_a_1.loc[0, 'a1,d2']+(1-p)*q_table_S1_a_2.loc[0, 'a1,d2']) * x[5] * x[6] + (p*q_table_S1_a_1.loc[0, 'a2,d1']+(1-p)*q_table_S1_a_2.loc[0, 'a2,d1']) * x[2] * x[7] + (p*q_table_S1_a_1.loc[0, 'a2,d1']+(1-p)*q_table_S1_a_2.loc[0, 'a2,d2']) * x[3] * x[7] + (p*q_table_S1_a_1.loc[0, 'a2,d2']+(1-p)*q_table_S1_a_2.loc[0, 'a2,d1']) * x[4] * x[7] + (p*q_table_S1_a_1.loc[0, 'a2,d2']+(1-p)*q_table_S1_a_2.loc[0, 'a2,d2']) * x[5] * x[7]


    y2 = (p*q_table_S1_d_1.loc[0, 'a1,d1']+(1-p)*q_table_S1_d_2.loc[0, 'a1,d1']) * x[2] * x[6] + (p*q_table_S1_d_1.loc[0, 'a1,d1']+(1-p)*q_table_S1_d_2.loc[0, 'a1,d2']) * x[3] * x[6] + (p*q_table_S1_d_1.loc[0, 'a1,d2']+(1-p)*q_table_S1_d_2.loc[0, 'a1,d1']) * x[4] * x[6] + (p*q_table_S1_d_1.loc[0, 'a1,d2']+(1-p)*q_table_S1_d_2.loc[0, 'a1,d2']) * x[5] * x[6] + (p*q_table_S1_d_1.loc[0, 'a2,d1']+(1-p)*q_table_S1_d_2.loc[0, 'a2,d1']) * x[2] * x[7] + (p*q_table_S1_d_1.loc[0, 'a2,d1']+(1-p)*q_table_S1_d_2.loc[0, 'a2,d2']) * x[3] * x[7] + (p*q_table_S1_d_1.loc[0, 'a2,d2']+(1-p)*q_table_S1_d_2.loc[0, 'a2,d1']) * x[4] * x[7] + (p*q_table_S1_d_1.loc[0, 'a2,d2']+(1-p)*q_table_S1_d_2.loc[0, 'a2,d2']) * x[5] * x[7]
    return x[0] + x[1] - y1 - y2

def constraint1(x):
    return x[0] - ((p*q_table_S1_a_1.loc[0, 'a1,d1']+(1-p)*q_table_S1_a_2.loc[0, 'a1,d1']) * x[6] + (p*q_table_S1_a_1.loc[0, 'a2,d1']+(1-p)*q_table_S1_a_2.loc[0, 'a2,d1']) * x[7])


def constraint2(x):
    return x[0] - ((p*q_table_S1_a_1.loc[0, 'a1,d1']+(1-p)*q_table_S1_a_2.loc[0, 'a1,d2']) * x[6] + (p*q_table_S1_a_1.loc[0, 'a2,d1']+(1-p)*q_table_S1_a_2.loc[0, 'a2,d2']) * x[7])

def constraint3(x):
    return x[0] - ((p*q_table_S1_a_1.loc[0, 'a1,d2']+(1-p)*q_table_S1_a_2.loc[0, 'a1,d1']) * x[6] + (p*q_table_S1_a_1.loc[0, 'a2,d2']+(1-p)*q_table_S1_a_2.loc[0, 'a2,d1']) * x[7])


def constraint4(x):
    return x[0] - ((p*q_table_S1_a_1.loc[0, 'a1,d2']+(1-p)*q_table_S1_a_2.loc[0, 'a1,d2']) * x[6] + (p*q_table_S1_a_1.loc[0, 'a2,d2']+(1-p)*q_table_S1_a_2.loc[0, 'a2,d2']) * x[7])

def constraint5(x):
    return x[1] - ((p*q_table_S1_d_1.loc[0, 'a1,d1']+(1-p)*q_table_S1_d_2.loc[0, 'a1,d1']) * x[2] + (p*q_table_S1_d_1.loc[0, 'a1,d1']+(1-p)*q_table_S1_d_2.loc[0, 'a1,d2']) * x[3] + (p*q_table_S1_d_1.loc[0, 'a1,d2']+(1-p)*q_table_S1_d_2.loc[0, 'a1,d1']) * x[4] + (p*q_table_S1_d_1.loc[0, 'a1,d2']+(1-p)*q_table_S1_d_2.loc[0, 'a1,d2']) * x[5])

def constraint6(x):
    return x[1] - ((p*q_table_S1_d_1.loc[0, 'a2,d1']+(1-p)*q_table_S1_d_2.loc[0, 'a2,d1']) * x[2] + (p*q_table_S1_d_1.loc[0, 'a2,d1']+(1-p)*q_table_S1_d_2.loc[0, 'a2,d2']) * x[3] + (p*q_table_S1_d_1.loc[0, 'a2,d2']+(1-p)*q_table_S1_d_2.loc[0, 'a2,d1']) * x[4] + (p*q_table_S1_d_1.loc[0, 'a2,d2']+(1-p)*q_table_S1_d_2.loc[0, 'a2,d2']) * x[5])


def constraint7(x):
    return x[2] + x[3] + x[4] + x[5] - 1


def constraint8(x):
    return x[6] + x[7] - 1

def choose_action(ACTIONS_S1, res):
    # This is how to choose an action
    if (np.random.uniform() > EPSILON):
        #or (state_actions.all() == 0)
        action_1 = np.random.choice(ACTIONS_S1)
        action_2 = np.random.choice(ACTIONS_S1)
    else:
        if (random.random() < (res[2]*res[6])):
            action_1 = 'a1,d1'
            action_2 = 'a1,d1'
        elif(random.random()< (res[2]*res[6]+res[3]*res[6])):
            action_1 = 'a1,d1'
            action_2 = 'a2,d1'
        elif(random.random()< (res[2]*res[6]+res[3]*res[6]+res[4]*res[6])):
            action_1 = 'a2,d1'
            action_2 = 'a1,d1'
        elif(random.random()< (res[2]*res[6]+res[3]*res[6]+res[4]*res[6]+res[5]*res[6])):
            action_1 = 'a2,d1'
            action_2 = 'a2,d1'
        elif(random.random()< (res[2]*res[6]+res[3]*res[6]+res[4]*res[6]+res[5]*res[6]+res[2]*res[7])):
            action_1 = 'a1,d2'
            action_2 = 'a1,d2'
        elif(random.random()< (res[2]*res[6]+res[3]*res[6]+res[4]*res[6]+res[5]*res[6]+res[2]*res[7]+res[3]*res[7])):
            action_1 = 'a1,d2'
            action_2 = 'a2,d2'
        elif(random.random()< (res[2]*res[6]+res[3]*res[6]+res[4]*res[6]+res[5]*res[6]+res[2]*res[7]+res[3]*res[7]+res[4]*res[7])):
            action_1 = 'a2,d2'
            action_2 = 'a1,d2'
        elif(random.random()< (res[2]*res[6]+res[3]*res[6]+res[4]*res[6]+res[5]*res[6]+res[2]*res[7]+res[3]*res[7]+res[4]*res[7]+res[5]*res[7])):
            action_1 = 'a2,d2'
            action_2 = 'a2,d2'
    return action_1, action_2


if __name__ == '__main__':
    q_table_S1_a_1 = build_q_table(N_STATES, ACTIONS_S1)
    q_table_S1_a_2 = build_q_table(N_STATES, ACTIONS_S1)
    q_table_S1_d_1 = build_q_table_d(N_STATES, ACTIONS_S1)
    q_table_S1_d_2 = build_q_table_d(N_STATES, ACTIONS_S1)
    for episode in range(0,150):
        action_S1_1, action_S1_2 = choose_action(ACTIONS_S1, res_process)
        R1_a_1, R1_d_1 = get_env_feedback1_a_1(action_S1_1)
        q_update_S1_a_1 = R1_a_1 - q_table_S1_a_1.loc[0, action_S1_1]
        q_table_S1_a_1.loc[0, action_S1_1] += ALPHA * q_update_S1_a_1
        q_update_S1_d_1 = R1_d_1 - q_table_S1_d_1.loc[0, action_S1_1]
        q_table_S1_d_1.loc[0, action_S1_1] += ALPHA * q_update_S1_d_1


        R1_a_2, R1_d_2= get_env_feedback1_a_2(action_S1_2)
        q_update_S1_a_2 = R1_a_2 - q_table_S1_a_2.loc[0, action_S1_2]
        q_table_S1_a_2.loc[0, action_S1_2] += ALPHA * q_update_S1_a_2
        q_update_S1_d_2 = R1_d_2 - q_table_S1_d_2.loc[0, action_S1_2]
        q_table_S1_d_2.loc[0, action_S1_2] += ALPHA * q_update_S1_d_2

        bounds = [(None, None)] * 2 + [(0, 1)] * 6
        x0 = np.array([13, -25, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
        res = minimize(objective, x0, bounds=bounds, method='SLSQP', constraints=[{'fun': constraint1, 'type': 'ineq'},
                                                                                  {'fun': constraint2, 'type': 'ineq'},
                                                                                  {'fun': constraint3, 'type': 'ineq'},
                                                                                  {'fun': constraint4, 'type': 'ineq'},
                                                                                  {'fun': constraint5, 'type': 'ineq'},
                                                                                  {'fun': constraint6, 'type': 'ineq'},
                                                                                  {'fun': constraint7, 'type': 'eq'},
                                                                                  {'fun': constraint8, 'type': 'eq'}
                                                                                  ])
        res_process = res.x
        print(res.fun, '\n', res.success, '\n', res.x)  # 输出最优值、求解状态、最优解
        results.append(res.x)


