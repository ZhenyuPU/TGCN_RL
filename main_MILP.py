import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Model import MILP_Model, MILP_Solver
from Utils.prediction_utils import Args


args = Args()

Solver = MILP_Solver.Solver(pred_method='TGCN_me')


# 开始执行MPC
res = Solver.implement_MPC(result_train_file='result/prediction_result/train', result_forecast_file=None, start_day=1, end_day=10, pred_horizon=24, horizon=24)