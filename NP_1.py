from scipy.optimize import minimize
import numpy as np
import pandas as pd

A1 = np.array([[8.0,10.9],
[9.9,9.5]])

D1 = np.array([[-26.5,-32.1],
[-37.4,-31.7]])
A2 = np.array([[8.6,12.9],
[10.5,6.5]])

D2 = np.array([[-24.6,-36.1],
[-35.3,-24.9]])
p = 0.7

def objective(x):
    y1 = (p*A1[0][0]+(1-p)*A2[0][0]) * x[2] * x[6] + (p*A1[0][0]+(1-p)*A2[0][1]) * x[3] * x[6] + (p*A1[0][1]+(1-p)*A2[0][0]) * x[4] * x[6] + (p*A1[0][1]+(1-p)*A2[0][1]) * x[5] * x[6] + (p*A1[1][0]+(1-p)*A2[1][0]) * x[2] * x[7] + (p*A1[1][0]+(1-p)*A2[1][1]) * x[3] * x[7] + (p*A1[1][1]+(1-p)*A2[1][0]) * x[4] * x[7] + (p*A1[1][1]+(1-p)*A2[1][1]) * x[5] * x[7]


    y2 = (p*D1[0][0]+(1-p)*D2[0][0]) * x[2] * x[6] + (p*D1[0][0]+(1-p)*D2[0][1]) * x[3] * x[6] + (p*D1[0][1]+(1-p)*D2[0][0]) * x[4] * x[6] + (p*D1[0][1]+(1-p)*D2[0][1]) * x[5] * x[6] + (p*D1[1][0]+(1-p)*D2[1][0]) * x[2] * x[7] + (p*D1[1][0]+(1-p)*D2[1][1]) * x[3] * x[7] + (p*D1[1][1]+(1-p)*D2[1][0]) * x[4] * x[7] + (p*D1[1][1]+(1-p)*D2[1][1]) * x[5] * x[7]
    return x[0] + x[1] - y1 - y2


def constraint1(x):
    return x[0] - ((p*A1[0][0]+(1-p)*A2[0][0]) * x[6] + (p*A1[1][0]+(1-p)*A2[1][0]) * x[7])


def constraint2(x):
    return x[0] - ((p*A1[0][0]+(1-p)*A2[0][1]) * x[6] + (p*A1[1][0]+(1-p)*A2[1][1]) * x[7])

def constraint3(x):
    return x[0] - ((p*A1[0][1]+(1-p)*A2[0][0]) * x[6] + (p*A1[1][1]+(1-p)*A2[1][0]) * x[7])


def constraint4(x):
    return x[0] - ((p*A1[0][1]+(1-p)*A2[0][1]) * x[6] + (p*A1[1][1]+(1-p)*A2[1][1]) * x[7])

def constraint5(x):
    return x[1] - ((p*D1[0][0]+(1-p)*D2[0][0]) * x[2] + (p*D1[0][0]+(1-p)*D2[0][1]) * x[3] + (p*D1[0][1]+(1-p)*D2[0][0]) * x[4] + (p*D1[0][1]+(1-p)*D2[0][1]) * x[5])

def constraint6(x):
    return x[1] - ((p*D1[1][0]+(1-p)*D2[1][0]) * x[2] + (p*D1[1][0]+(1-p)*D2[1][1]) * x[3] + (p*D1[1][1]+(1-p)*D2[1][0]) * x[4] + (p*D1[1][1]+(1-p)*D2[1][1]) * x[5])


def constraint7(x):
    return x[2] + x[3] + x[4] + x[5] - 1


def constraint8(x):
    return x[6] + x[7] - 1


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
pro = res.x
pro = np.expand_dims(pro, axis=0)
print(res.fun, '\n', res.success, '\n', res.x)  # 输出最优值、求解状态、最优解
