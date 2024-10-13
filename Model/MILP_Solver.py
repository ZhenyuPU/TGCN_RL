"""
    MPC, SP, Perfect information MILP Solver Programming
"""
from Model.MILP_Model import HIES_MILP
import gurobipy as gp
from  gurobipy import GRB
from Utils.HIES_options import HIES_Options
from Model.TGCN_me import TGCN as TGCN_me
from Model.TGCN_origin import TGCN as TGCN_origin
import numpy as np
import os
import json
import torch
import torch.nn as nn
import time
from Model.handler import prediction_train, get_results
from Utils.prediction_utils import Args

import pandas as pd





 
class Solver:
    def __init__(self, pred_method):
        self.args = Args()
        self.Model_MILP = HIES_MILP()
        self.env_options = HIES_Options()
        self.T = self.env_options.T
        self.data = pd.read_csv('datasets/HIES_data.csv').values

        self.train_data = self.data[:int(self.args.train_ratio * len(self.data))]
        self.valid_data = self.data[int(self.args.train_ratio * len(self.data)):int((self.args.train_ratio + self.args.valid_ratio) * len(self.data))]
        self.test_data = self.data[int((self.args.train_ratio + self.args.valid_ratio) * len(self.data)):]

        units = self.data.shape[1]
        

        if pred_method == 'TGCN_me':
            self.model = TGCN_me(units=units, stack_cnt=self.args.stack_cnt, time_step=self.args.window_size, horizon=self.args.horizon)
        elif pred_method == 'TGCN_origin':
            self.model = TGCN_origin(unit=units, stack_cnt=self.args.stack_cnt, time_step=self.args.window_size, horizon=self.args.horizon)
    


    def MPC_prediction(self, result_train_file, result_forecast_file, train=True, data=None, args=None, horizon=None):
        """
            To train prediction model or get prediction results
            Prediction horizon + Optimize
            prediction:
                Electricity price
                Solar radiation
                electricity demand
                heating demand
                cooling demand
                T_a ambient temperature
        """
        if train:
            if not os.path.exists(result_train_file):
                os.makedirs(result_train_file)
            performance_metrics, normalize_statistic = prediction_train(self.train_data, self.valid_data, self.model, result_train_file, args)
            return None
        else:
            # 获取预测值，针对特定区间
            batch_size = 1
            forecast_data = get_results(data, args, result_train_file, result_forecast_file, batch_size, horizon)
            return forecast_data

    # 利用预测数据进行优化
    def run_MPC_model(self, selected_day, time_start=0, time_end=24, pred_res = None, forecast_data=None):
        params = {}
        params['time_start'] = time_start
        params['time_end'] = time_end

        # load_data(method='PRED', pred_method=pred_method)
        PRED_model = gp.Model('PRED_model')
        PRED_result = {}
        PRED_model, PRED_result, obj =  self.Model_MILP.model(PRED_model, PRED_result, params, method='PRED', S=1, selected_day=selected_day, pred_res = pred_res, forecast_data=forecast_data)

        PRED_model.setObjective(obj, sense = GRB.MINIMIZE)
        PRED_model.optimize()   ### 开始问题求解

        PRED_result['ObjVal'] = PRED_model.ObjVal  ### 获取求解目标函数值

        if  PRED_model.status == gp.GRB.OPTIMAL:
            print("HEMS operation: optimal solution found ! \n")
            print(f"Optimal objective value: {round(PRED_model.ObjVal, 2)} \n")
            # Obtain all decision and state values
            decision_result = self.save_results(PRED_model, PRED_result, Saved = False,  file_name=None)

        elif PRED_model.status == gp.GRB.INFEASIBLE:
            print("PRED_model is infeasible!\n")

        elif PRED_model.status == gp.GRB.UNBOUNDED:
            print("PRED_model is unbounded!\n")

        else:
            print(f"Optimization ended with status: {PRED_model.status}\n" )
        
        time_pred = PRED_model.Runtime
        return PRED_model.ObjVal, PRED_result, decision_result




    # 将预测优化结果在真实系统中执行(确定性规划)
    # 预测C，使用C，直到到达优化区间N
    # N = 24， C <= N
    def Appl_MPC_model(self, selected_day, pred_horizon, forecast_data=None):
        env_options = HIES_Options()
        res = []
        # 判断是否整除优化时间范围，求出循环次数
        if self.T % pred_horizon != 0:
            N = int(self.T / pred_horizon) + 1
        else:
            N = int(self.T / pred_horizon)
        
        for i in range(N):
            result = None
            if i == N-1 and self.T % pred_horizon != 0:
                time_rest = self.T % pred_horizon  # 剩余没有优化的时刻
                pred_horizon = time_rest
                time_start = self.T-time_rest
                time_end = self.T
            else:
                time_start = i * pred_horizon
                time_end = (i + 1) * pred_horizon

            _, PRED_res, decisions = self.run_MPC_model(selected_day, time_start=time_start, time_end=time_end, pred_res = result, forecast_data=forecast_data)
            res.append(decisions)   # Store all decisions
            # 在实际模型(确定性规划)中求解实际的reward, cost, violation
            opt_model = gp.Model('MPC_Opt')
            opt_res = {}
            # 将预测结果用在实际系统中
            opt_model, opt_res, obj, obj_decs = self.Model_MILP.model(opt_model, opt_res, method='OPT', S=1, selected_day=selected_day)
            P_g_buy_pred   = PRED_res["P_g_buy"]
            P_g_sell_pred  = PRED_res["P_g_sell"]
            m_B_pred       = PRED_res["m_B"]
            
            # add constraints
            opt_model.addConstrs(obj_decs['P_g_buy'][t]  == P_g_buy_pred[t]  for t in range(self.T) )
            opt_model.addConstrs(obj_decs['P_g_sell'][t] == P_g_sell_pred[t] for t in range(self.T) )
            opt_model.addConstrs(obj_decs['m_B'][t]      == m_B_pred[t]      for t in range(self.T) )

            opt_model.setObjective(obj, sense = GRB.MINIMIZE)
            opt_model.setParam('OutputFlag', 0)
            opt_model.optimize()

            if opt_model.status == gp.GRB.OPTIMAL:
                # obtain the optimal value
                opt_res['ObjVal'] = opt_model.ObjVal
                print('Optimal Operation Found !\n')
                print(f'Optimal Operation Cost: {round(opt_model.ObjVal, 2)}\n')
                result = self.save_results(opt_model, opt_res, Saved = False, file_name=None)
            elif opt_model.status == gp.GRB.INFEASIBLE:
                print("HEMS Opt model is infeasible!\n")
            elif opt_model.status == gp.GRB.UNBOUNDED:
                print("HEMS Opt model is unbounded!\n")
            else:
                print( f"Optimization ended with status: {opt_model.status}\n")

        # 处理res列表里面的数据，包括每一次将预测结果用于系统的数据，列表里面每一个元素是一个字典，字典的关键词都一样的
        keys = list(res[0].keys())
        res_final = {
            k: np.array([res_t[k] for res_t in res]).flatten() 
            for k in keys
        }      
        return res_final
    


    def SP_model(self):
        pass



    def OPT_model(self):
        pass




    def save_results(self, model, result, Saved, file_name):
        for var in model.getVars():

            var_name  = var.VarName.split("[")[0]
            var_index = var.VarName.split("[")[1]

            if "," in var_index:
                if len( var_index.split(",") ) == 3:
                    var_fac    =  int( var_index.split(",")[0] )
                    var_rol    =  int( var_index.split(",")[1] )
                    var_col    =  int( var_index.split(",")[2][:-1] )
                    result[var_name][var_fac, var_rol, var_col] = var.X

                elif len( var_index.split(",") ) == 2:
                    var_rol    =  int( var_index.split(",")[0] )
                    var_col    =  int( var_index.split(",")[1][:-1] )
                    result[var_name][var_rol, var_col] = var.X
            else:
                var_col   = int(var_index[:-1])
                result[var_name][var_col] = var.X 

        if Saved:
            np.savez(f'result/{file_name}', **result)

        return result #把优化求解结果返回
    


    def implement_MPC(self, result_train_file, result_forecast_file, start_day, end_day, pred_horizon, horizon):
        my_result = [ 0 ] * (end_day - start_day )  #创建字典存储结果
        for idx, selected_day in enumerate( range(start_day, end_day) ):
            print(f'Run HIES MPC on day: {selected_day}\n')
            
            forecast_data = self.MPC_prediction(result_train_file=result_train_file, result_forecast_file=result_forecast_file, train=False, args=self.args, data=self.data[(selected_day - 1) * 24: selected_day * 24, :], horizon=horizon)

            my_result[idx] = self.Appl_MPC_model(selected_day=selected_day, pred_horizon=pred_horizon, forecast_data=forecast_data)   # 利用前一天数据预测
        
        return my_result


