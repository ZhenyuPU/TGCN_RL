"""
    MILP in the MPC fashion

    States:
        PV power: deterministic
        Price: deterministic
        T: predicted
        Demand: deterministic
        i_t
        E^{el}_t
        C_{t}

    Decision variables:
        P^{el}_t
        P^{el, l}_t
    
    Constraints:
        &E_{t+1} = E_{t} + E^{el}_t - E^{dem}_t \\
        &E^{el}_t = f(P^{el}_t, T) \\
        &T_{t+1} = g_{T}T_{t} + g_{P}P^{el}_t + g_{k} \\
        &0 \leq P^{el}_t \leq P^{C, max} \\
        &T_{t} \leq T^{max} \\
        &i^{1}_t = f^{(i, 1)}(P^{el, 1}(P^{el}_t), T_t),& i^1_t \geq 0 \\
        &C_{t+1} = max(C_t + \delta_t(i^1_t - i^{nom}), 0) \\
        &P^{el, AC}_t = m^{AC}P^{el}_t + k^{AC} \\
        &P^{grid}_t + P^{PV}_t \geq P^{el, AC}_t \\
    
    Objective:
        min \sum_{t=0}^{N-1}g(P^{el}_t)
"""

import numpy as np
import gurobipy as gp
from  gurobipy import GRB
import pandas as pd
from Utils.HIES_options import HIES_Options
from Model.TGCN_me import TGCN as TGCN_me
from Model.TGCN_origin import TGCN as TGCN_origin
import os
import json
import torch
import torch.nn as nn
import time

from Utils.prediction_utils import Args

from handler import prediction_train, get_results, test

class HIES_MILP:
    def __init__(self, penalty_factor=10) -> None:
        self.S =  30

        self.env_options = HIES_Options

        self.penalty_factor = penalty_factor


        #! State definition
        self.T_a = self.env_options.T_a    # ambient temperature

        self.T =  self.env_options.T
        self.DELTA  = self.env_options.DELTA

        self.P_solar_gen   =  self.env_options.P_solar_gen   # 光伏发电数据
        self.P_solar_heat  =  self.env_options.P_solar_heat  # 光伏集热器数据

        self.lambda_b      =  self.env_options.lambda_b      # 电力市场买电价格
        self.lambda_s      =  self.env_options.lambda_s      # 电力市场卖电价格
        self.lambda_B      =  self.env_options.lambda_B      # 氢市场买氢价格

        self.lambda_ED    =   self.env_options.lambda_ED  * penalty_factor     # 电需求未满足惩罚因子
        self.lambda_HD    =   self.env_options.lambda_HD  * penalty_factor     # 热水需求未满足的惩罚因子
        self.lambda_CD    =   self.env_options.lambda_CD  * penalty_factor     # 冷水需求未满足的惩罚因子
        
        self.Q_ED         =   self.env_options.Q_ED             # 用户电需求
        self.Q_HD         =   self.env_options.Q_HD             # 用户热水需求
        self.Q_CD         =   self.env_options.Q_CD             # 用户冷水需求    
        self.time_sp      =   0                             # SP运行时间
        self.time_opt     =   0                             # opt运行时间 



    def MPC_prediction(self, pred_method):
        """
            To get prediction results
            Prediction horizon + Optimize
            prediction:
                Electricity price
                Solar radiation
                electricity demand
                heating demand
                cooling demand
                T_a ambient temperature
        """
        if pred_method == 'TGCN_me':
            model = TGCN_me()
        elif pred_method == 'TGCN_origin':
            model = TGCN_origin()
        args = Args()
        performance_metrics, normalize_statistic = prediction_train(train_data, valid_data, model, result_file)
        get_results(data, args, result_train_file, result_forecast_file)
            

    # Load prediction data
    def load_data(self, method=None, pred_method='LFC-DC'):
        if method == 'PRED':
            electricity_price    = np.load(f'Forecast/cases/electricity_price/Case_{pred_method}_electricity_price.npz', allow_pickle=True)
            solar_radiation      = np.load(f'Forecast/cases/solar_radiation/Case_{pred_method}_solar_radiation.npz', allow_pickle=True)['pred']
            electricity_demand   = np.load(f'Forecast/cases/electricity_demand/Case_{pred_method}_electricity_demand.npz', allow_pickle=True)
            heating_demand       = np.load(f'Forecast/cases/heating_demand/Case_{pred_method}_heating_demand.npz', allow_pickle=True)
            cooling_demand       = np.load(f'Forecast/cases/cooling_demand/Case_{pred_method}_cooling_demand.npz', allow_pickle=True)
            T_a  # TODO

            self.lambda_b     = electricity_price['pred']                           # 电价预测值，从day=1开始
            self.P_solar_gen  = solar_radiation * self.env_options.S_solar_gen * self.env_options.eta_solar_gen /1000
            self.P_solar_heat = solar_radiation * self.env_options.S_solar_heat * self.env_options.eta_solar_heat/1000
            self.Q_ED         = electricity_demand['pred']                          # 电负荷需求预测值
            self.Q_HD         = heating_demand['pred']                              # 热负荷需求预测值
            self.Q_CD         = cooling_demand['pred']                              # 冷负荷需求预测值

            self.lambda_ED  = self.penalty_factor * np.mean(self.lambda_b, axis = 0)     # 电需求未满足惩罚因子
            self.lambda_CD  = self.penalty_factor * np.mean(self.lambda_b, axis = 0)     # 热水需求未满足惩罚因子
            self.lambda_HD  = self.penalty_factor * np.mean(self.lambda_b, axis = 0)     # 冷水需求未满足惩罚因子


        else:
            self.lambda_b = self.env_options.lambda_b                           # 电价预测值，从day=1开始
            self.Q_ED     = self.env_options.Q_ED                               # 电负荷需求预测值
            self.Q_HD     = self.env_options.Q_HD                               # 热负荷需求预测值
            self.Q_CD     = self.env_options.Q_CD                               # 冷负荷需求预测值

            self.lambda_ED    =   self.env_options.lambda_ED  * self.penalty_factor     # 电需求未满足惩罚因子
            self.lambda_HD    =   self.env_options.lambda_HD  * self.penalty_factor     # 热水需求未满足的惩罚因子
            self.lambda_CD    =   self.env_options.lambda_CD  * self.penalty_factor     # 冷水需求未满足的惩罚因子
    

    def model(self, model, result, params=None, method='PRED', S=1, selected_day=None, T = 24, pred_res = None):
        if method == 'PRED':
            time_start = params['time_start']
            time_end = params['time_end']
            self.T = time_end - time_start
        else:
            self.T = T
        ### 定义决策变量
        self.P_g_buy    = model.addVars(self.T,  lb = 0,  ub = GRB.INFINITY,  name="P_g_buy",  vtype=GRB.CONTINUOUS )  # 与电网买卖电量[单位：kW]
        self.P_g_sell   = model.addVars(self.T,  lb = 0,  ub = GRB.INFINITY,  name="P_g_sell",  vtype=GRB.CONTINUOUS )  # 与电网买卖电量[单位：kW]

        self.m_B     = model.addVars(self.T,  lb = 0,  ub = GRB.INFINITY,  name="m_B",  vtype=GRB.CONTINUOUS )              # 氢市场买氢量[单位：kg]

        self.u_ely    = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="u_ely",  vtype=GRB.CONTINUOUS   )           # 电解池输入功率[单位：kW]
        self.u_fc    = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="u_fc",  vtype=GRB.CONTINUOUS   )           # 燃料电池输出功率[单位：kW]

        
        self.P_bssc  = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="P_bssc",  vtype=GRB.CONTINUOUS )           # 电池充电功率[单位：kW]
        self.P_bssd  = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="P_bssd",  vtype=GRB.CONTINUOUS )           # 电池放电功率[单位：kW]
        
        self.g_AC    = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="g_AC",  vtype=GRB.CONTINUOUS   )           # 吸收式制冷机输入功率
        self.g_tesc  = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="g_tesc",  vtype=GRB.CONTINUOUS )           # 储热水罐输入功率
        self.g_tesd  = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="g_tesd",  vtype=GRB.CONTINUOUS )           # 储热水罐输出功率

        self.q_cssc  = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="q_cssc",  vtype=GRB.CONTINUOUS )           # 储冷水罐输入功率
        self.q_cssd  = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="q_cssd",  vtype=GRB.CONTINUOUS )           # 储冷水罐输出功率

        self.P_CO    = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="P_CO",  vtype=GRB.CONTINUOUS )             # 氢压缩罐耗能
        self.g_FC    = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="g_FC",  vtype=GRB.CONTINUOUS )             # 燃料电池产热量
        self.g_EL    = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="g_EL",  vtype=GRB.CONTINUOUS )             # 电解池产热量

        #! States
        self.S_bss  = model.addVars(S, self.T + 1,  lb = 0,  ub = GRB.INFINITY,  name="S_bss",  vtype=GRB.CONTINUOUS )        # 电储能状态

        self.S_hss  = model.addVars(S, self.T + 1,  lb = 0,  ub = GRB.INFINITY,  name="S_hss",  vtype=GRB.CONTINUOUS )        # 储氢罐状态
        self.T_ely  = model.addVars(S, self.T + 1,  lb = 0,  ub = GRB.INFINITY,  name="T_ely",  vtype=GRB.CONTINUOUS )        # 电解槽温度
        self.T_tank = model.addVars(S, self.T + 1,  lb = 0,  ub = GRB.INFINITY,  name="T_tank",  vtype=GRB.CONTINUOUS )        # Hydrogen Tank Temperature
        self.C      = model.addVars(S, self.T + 1,  lb = 0,  ub = GRB.INFINITY,  name="C",  vtype=GRB.CONTINUOUS )        # C

        self.S_tes  = model.addVars(S, self.T + 1,  lb = 0,  ub = GRB.INFINITY,  name="S_tes",  vtype=GRB.CONTINUOUS )        # 储热水罐状态
        self.S_css  = model.addVars(S, self.T + 1,  lb = 0,  ub = GRB.INFINITY,  name="S_css",  vtype=GRB.CONTINUOUS )        # 储冷水罐状态

        self.Q_ED_pos = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="Q_ED_pos",  vtype=GRB.CONTINUOUS )       # 未满足电需求
        self.Q_HD_pos = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="Q_HD_pos",  vtype=GRB.CONTINUOUS )       # 未满足热需求
        self.Q_CD_pos = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="Q_CD_pos",  vtype=GRB.CONTINUOUS )       # 未满足热需求
       
       ### 定义字典存储求解结果
        result['P_g_buy']  = np.full( self.T, np.nan)
        result['P_g_sell'] = np.full( self.T, np.nan)
        result['m_B'   ]   = np.full( self.T, np.nan)

        result['u_ely'  ] = np.full( (S, self.T), np.nan)
        result['u_fc'  ] = np.full( (S, self.T), np.nan)

        result['P_bssc'] = np.full( (S, self.T), np.nan)
        result['P_bssd'] = np.full( (S, self.T), np.nan)
        
        result['g_AC'  ] = np.full( (S, self.T), np.nan)

        result['g_tesc'] = np.full( (S, self.T), np.nan)
        result['g_tesd'] = np.full( (S, self.T), np.nan)

        result['q_cssc'] = np.full( (S, self.T), np.nan)
        result['q_cssd'] = np.full( (S, self.T), np.nan)

        result['P_CO']   = np.full( (S, self.T), np.nan)
        result['g_FC']   = np.full( (S, self.T), np.nan)
        result['g_EL']   = np.full( (S, self.T), np.nan)

        result['Q_ED_pos']   = np.full( (S, self.T), np.nan)
        result['Q_HD_pos']   = np.full( (S, self.T), np.nan)
        result['Q_CD_pos']   = np.full( (S, self.T), np.nan)

        result['S_bss']   = np.full( (S, self.T + 1), np.nan)
        result['S_hss']   = np.full( (S, self.T + 1), np.nan)
        result['S_tes']   = np.full( (S, self.T + 1), np.nan)
        result['S_css']   = np.full( (S, self.T + 1), np.nan)
        result['T_ely']   = np.full( (S, self.T + 1), np.nan)
        result['T_tank']  = np.full( (S, self.T + 1), np.nan)
        result['C']       = np.full( (S, self.T + 1), np.nan)
 

        ### 初始化储能设备状态
        if pred_res:
            for s in range(S):
                model.addConstr(self.S_bss[s, 0] ==  pred_res['S_bss'][-1]  )

                model.addConstr(self.S_hss[s, 0] ==  pred_res['S_hss'][-1]  )
                model.addConstr(self.T_ely[s, 0] == pred_res['T_ely'][-1])
                model.addConstr(self.T_tank[s, 0] == pred_res['T_tank'][-1])
                model.addConstr(self.C[s, 0] == pred_res['C'][-1])

                model.addConstr(self.S_tes[s, 0] ==  pred_res['S_tes'][-1]  )
                model.addConstr(self.S_css[s, 0] == pred_res['S_css'][-1]  )
        else:
            for s in range(S):
                model.addConstr(self.S_bss[s, 0] ==  self.env_options.S_bss_init  )

                model.addConstr(self.S_hss[s, 0] ==  self.env_options.S_hss_init  )
                model.addConstr(self.T_ely[s, 0] == self.env_options.T_ely_init)
                model.addConstr(self.T_tank[s, 0] == self.env_options.T_tank_init)
                model.addConstr(self.C[s, 0] == self.env_options.C_init)

                model.addConstr(self.S_tes[s, 0] ==  self.env_options.S_tes_init  )
                model.addConstr(self.S_css[s, 0] ==  self.env_options.S_css_init  )

        ### 电力市场和氢市场运行约束
        for t in range(self.T):
            model.addConstr( self.P_g_buy[t]  <=  self.env_options.P_g_max   )
            model.addConstr( self.P_g_sell[t] <=  self.env_options.P_g_max   )


        ### 设备运行物理约束
        for s in range(S):    
            for t in range(self.T):
                #! Electrolyzer
                P_ely = self.u_ely[s, t] * self.env_options.q1 + self.env_options.q0
                model.addConstr( P_ely   <= self.env_options.P_ely_high   )
                model.addConstr( P_ely   >= self.env_options.P_ely_low   )
                #! Fuel cell
                model.addConstr( self.u_fc[s, t]   <= self.env_options.u_high   )
                model.addConstr( self.u_fc[s, t]   >= self.env_options.u_low   )
                # tank pressure
                p_t = (self.env_options.b0 + self.env_options.b1 * self.T_tank[s, t]) * (self.S_hss[s, t] / self.env_options.V_tank) / (10**6)
                model.addConstr( p_t   <= self.env_options.p_high   )
                model.addConstr( p_t   >= self.env_options.p_low   )



                # Battery
                model.addConstr( self.P_bssc[s, t] <= self.env_options.p_bssc_max )
                model.addConstr( self.P_bssd[s, t] <= self.env_options.p_bssd_max )
                # AC
                model.addConstr( self.g_AC[s, t]   <= self.env_options.g_AC_max   )
                # Thermal energy storage
                model.addConstr( self.g_tesc[s, t] <= self.env_options.g_tesc_max )
                model.addConstr( self.g_tesd[s, t] <= self.env_options.g_tesd_max )
                # Chilled water storage
                model.addConstr( self.q_cssc[s, t] <= self.env_options.q_cssc_max )
                model.addConstr( self.q_cssd[s, t] <= self.env_options.q_cssd_max )
        
        ### 设备运行动态特性方程
        for s in range(S):
            for t in range(self.T):
                # Battery
                model.addConstr( self.S_bss[s, t + 1] == self.S_bss[s, t] + ( self.P_bssc[s, t] * self.env_options.eta_bssc - self.P_bssd[s, t]/self.env_options.eta_bssd ) * self.DELTA )
                #! Hydrogen


                # Electrolyzer
                u_ely = self.u_ely[s, t]   # DC Power
                # volume rate of hydrogen
                if u_ely > self.env_options.P_ely_low and u_ely < self.env_options.P_nom_ely:
                    v_ely = self.env_options.z1 * self.T_ely[s, t] + self.env_options.z0 + self.env_options.z_low * (u_ely - self.env_options.P_nom_ely)
                elif u_ely > self.env_options.P_nom_ely and u_ely <= self.env_options.P_ely_high:
                    v_ely = self.env_options.z1 * self.T_ely[s, t] + self.env_options.z0 + self.env_options.z_high * (u_ely - self.env_options.P_nom_ely)
                else:
                    v_ely = 0


                # Overload counter model
                T_ely = self.T_ely[s, t]
                if self.env_options.P_ely_low < u_ely <= self.env_options.P_nom_ely/4:
                    P_ely_1 = u_ely
                elif self.env_options.P_nom_ely/4 < u_ely <= self.env_options.P_nom_ely:
                    P_ely_1 = self.env_options.P_nom_ely / 4
                elif self.env_options.P_nom_ely < u_ely <= self.env_options.P_ely_high:
                    P_ely_1 = u_ely / 4
                else:
                    P_ely_1 = 0
                # Current
                if self.env_options.P_ely_low < P_ely_1 <= self.env_options.P_nom_ely_1:
                    i_ely = self.env_options.h1 * T_ely + self.env_options.h0 + self.env_options.h_low * (P_ely_1 - self.env_options.P_nom_ely_1)
                elif self.env_options.P_nom_ely_1 < P_ely_1 <= self.env_options.P_ely_high:
                    i_ely = self.env_options.h1 * T_ely + self.env_options.h0 + self.env_options.h_high * (P_ely_1 - self.env_options.P_nom_ely_1)
                else:
                    i_ely = 0

                
                # Storage Tank
                v_cdg = self.env_options.alpha * v_ely

                # Fuel cell
                # Current of FC
                if self.env_options.u_low < self.u_fc[s, t] <= self.env_options.u_bp_fc:
                    i_fc = self.env_options.s1 * self.u_fc[s, t]
                elif self.env_options.u_bp_fc < self.u_fc[s, t] <= self.env_options.u_high:
                    i_fc = self.env_options.s2 * (self.u_fc[s, t] - self.env_options.u_bp_fc) + self.env_options.i_bp_fc
                else:
                    i_fc = 0
                v_fc = self.env_options.c * i_fc
                
                model.addConstr( self.S_hss[s, t + 1] == self.S_hss[s, t] + self.env_options.rho * (v_cdg - v_fc) )
                #! temperature of eleyctrolyzer
                model.addConstr( self.T_ely[s, t + 1] == self.T_ely[s, t] * self.env_options.j1 + self.env_options.j2 * self.u_ely[s, t] + self.env_options.j0)
                #! Tank temperature
                model.addConstr( self.T_tank[s, t + 1] == self.T_tank[s, t] * self.env_options.g0 + self.env_options.g1 * self.T_a[selected_day, t])
                #! Current counter dynamics
                model.addConstr( self.C[s, t + 1] == max(0, self.C[s, t] + self.DELTA * (i_ely - self.env_options.i_nom_ely)))
                


                model.addConstr( self.S_tes[s, t + 1] == self.S_tes[s, t] + ( self.g_tesc[s, t] * self.env_options.eta_tesc - self.g_tesd[s, t]/self.env_options.eta_tesd ) * self.DELTA )
                model.addConstr( self.S_css[s, t + 1] == self.S_css[s, t] + ( self.q_cssc[s, t] * self.env_options.eta_cssc - self.q_cssd[s, t]/self.env_options.eta_cssd ) * self.DELTA )
        
        ### 供需平衡约束
        for s in range(S):
            for t in range(self.T):
                # model.addConstr( self.P_CO[s, t] == self.env_options.k_CO * self.env_options.k_EL * self.P_EL[s, t] )  #氢压缩机耗电量)
                model.addConstr( self.P_CO[s, t] == 0 )  #氢压缩机耗电量)
                model.addConstr( self.g_FC[s, t] == self.env_options.eta_FC_rec * (1 - self.env_options.eta_FC) * self.u_fc[s, t]/self.env_options.eta_FC )   ### 燃料电池产热量

                model.addConstr( self.g_EL[s, t] == self.env_options.eta_EL_rec * (1 - self.env_options.eta_EL) * self.u_ely[s, t] )   ### 电解池产热量

                model.addConstr( self.Q_ED_pos[s, t] 
                                >= 
                                (self.P_g_sell[t] + self.u_ely[s, t] + self.P_CO[s, t] + 
                                 self.P_bssc[s, t] + self.Q_ED[s, t]) 
                                 - (self.P_g_buy[t] +  self.P_solar_gen[s, t] + self.u_fc[s, t] + self.P_bssd[s, t]) )  if selected_day is None else model.addConstr( self.Q_ED_pos[s, t] >= 
                                 (self.P_g_sell[t] + self.u_ely[s, t] + self.P_CO[s, t] + self.P_bssc[s, t] + self.Q_ED[selected_day, t]) - (self.P_g_buy[t] +  self.P_solar_gen[selected_day, t] + self.u_fc[s, t] + self.P_bssd[s, t]) )### 电能量供需平衡方程

                model.addConstr( self.Q_HD_pos[s, t] >= 
                                (self.g_tesc[s, t] + self.Q_HD[s, t]) - (self.g_FC[s, t] + self.g_EL[s, t] - self.g_AC[s, t] +  self.g_tesd[s, t] + self.P_solar_heat[s, t]) ) if selected_day is None else  model.addConstr( self.Q_HD_pos[s, t] >= 
                                (self.g_tesc[s, t] + self.Q_HD[s, t]) 
                                - (self.g_FC[s, t] + self.g_EL[s, t] - self.g_AC[s, t] +  self.g_tesd[s, t] + self.P_solar_heat[selected_day, t]) ) ### 热量供需平衡方程
                
                model.addConstr( self.Q_CD_pos[s, t] >= 
                                (self.q_cssc[s, t] + self.Q_CD[s, t]) - (self.g_AC[s, t] * self.env_options.eta_AC  + self.q_cssd[s, t]) )  if selected_day is None else model.addConstr( self.Q_CD_pos[s, t] >= (self.q_cssc[s, t] + self.Q_CD[selected_day, t]) - (self.g_AC[s, t] * self.env_options.eta_AC  + self.q_cssd[s, t]) ) ### 冷平衡方程
        
        if selected_day is None:
            obj = (   gp.quicksum( self.lambda_b[s, t] * self.P_g_buy[t]  for s in range(S) for t in range(self.T) ) 
                     - gp.quicksum( self.lambda_s[s, t] * self.P_g_sell[t] for s in range(self.S) for t in range(self.T) ) 
                     + gp.quicksum( self.lambda_B[s, t] * self.m_B[t]      for s in range(self.S) for t in range(self.T) )
                   ) 
        else:
            obj = (   gp.quicksum( self.lambda_b[selected_day, t] * self.P_g_buy[t]  for s in range(S) for t in range(self.T) ) 
                     - gp.quicksum( self.lambda_s[selected_day, t] * self.P_g_sell[t] for s in range(self.S) for t in range(self.T) ) 
                     + gp.quicksum( self.lambda_B[selected_day, t] * self.m_B[t]      for s in range(self.S) for t in range(self.T) )
                   ) 
            
        obj = obj + (    gp.quicksum( self.lambda_ED[t] * self.Q_ED_pos[s, t] for s in range(S) for t in range(self.T)  )
                                +  gp.quicksum( self.lambda_HD[t] * self.Q_HD_pos[s, t] for s in range(S) for t in range(self.T)  )
                                +  gp.quicksum( self.lambda_CD[t] * self.Q_CD_pos[s, t] for s in range(S) for t in range(self.T)  )
                            )
        return model, result, obj
    

    #TODO 检查求解优化的范围，看一下之前做的平台，查看优化的是一天的还是多少天的，如何选择优化的时间段的，是以一天最优来看的，所以预测范围应该是一天
    
    def run_MPC_model(self, selected_day, pred_horizon=24, time_start=0, time_end=24, pred_method='TGCN_me', pred_res = None):
        params = {}
        params['time_start'] = time_start
        params['time_end'] = time_end

        self.load_data(method='PRED', pred_method=pred_method)
        self.PRED_model = gp.Model('PRED_model')
        self.PRED_result = {}
        self.PRED_model, self.PRED_result, self.obj =  self.model(self.PRED_model, self.PRED_result, params, method='PRED', S=1, selected_day=selected_day, pred_res = None)

        self.PRED_model.setObjective(self.obj, sense = GRB.MINIMIZE)
        self.PRED_model.optimize()   ### 开始问题求解

        self.PRED_result['ObjVal'] = self.PRED_model.ObjVal  ### 获取求解目标函数值

        if  self.PRED_model.status == gp.GRB.OPTIMAL:
            print("HEMS operation: optimal solution found ! \n")
            print(f"Optimal objective value: {round(self.PRED_model.ObjVal, 2)} \n")
            # Obtain all decision and state values
            decision_result = self.save_results(self.PRED_model, self.PRED_result, Saved = False,  file_name=None)

        elif self.PRED_model.status == gp.GRB.INFEASIBLE:
            print("PRED_model is infeasible!\n")

        elif self.PRED_model.status == gp.GRB.UNBOUNDED:
            print("PRED_model is unbounded!\n")

        else:
            print(f"Optimization ended with status: {self.PRED_model.status}\n" )
        
        self.time_pred = self.PRED_model.Runtime
        return self.PRED_model.ObjVal, self.PRED_result, decision_result
    


    # 预测C，使用C，直到到达优化区间N
    # N = 24， C <= N
    def run_model(self, selected_day, program, pred_horizon):
        res = []
        # 判断是否整除优化时间范围，求出循环次数
        if self.env_options.T % pred_horizon != 0:
            N = int(self.env_options.T / pred_horizon) + 1
        else:
            N = self.env_options.T / pred_horizon
        
        for i in range(N):
            result = None
            if i == N-1 and self.env_options.T % pred_horizon != 0:
                time_rest = self.env_options.T % pred_horizon  # 剩余没有优化的时刻
                pred_horizon = time_rest
                time_start = self.env_options.T-time_rest
                time_end = self.env_options.T-1
            else:
                time_start = i * pred_horizon
                time_end = (i + 1) * pred_horizon

            _, PRED_res, decisions = self.run_MPC_model(selected_day, pred_horizon, time_start=time_start, time_end=time_end, pred_res = result)
            res.append(decisions)   # Store all decisions
            # 在实际模型(确定性规划)中求解实际的reward, cost, violation
            opt_model = gp.Model('MPC_Opt')
            opt_res = {}
            # 将预测结果用在实际系统中
            opt_model, opt_res, obj = self.model(opt_model, opt_res, method='OPT', S=1, selected_day=selected_day)
            P_g_buy_pred   = PRED_res["P_g_buy"]
            P_g_sell_pred  = PRED_res["P_g_sell"]
            m_B_pred       = PRED_res["m_B"]
          
            # add constraints
            opt_model.addConstrs(self.P_g_buy[t]  == P_g_buy_pred[t]  for t in range(self.T) )
            opt_model.addConstrs(self.P_g_sell[t] == P_g_sell_pred[t] for t in range(self.T) )
            opt_model.addConstrs(self.m_B[t]      == m_B_pred[t]      for t in range(self.T) )

            opt_model.setObjective(self.obj, sense = GRB.MINIMIZE)
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
                print( f"Optimization ended with status: {self.Opt_model.status}\n")

        # 处理res列表里面的数据，包括每一次将预测结果用于系统的数据，列表里面每一个元素是一个字典，字典的关键词都一样的
        keys = list(res[0].keys())
        res_final = {
            k: np.array([res_t[k] for res_t in res]).flatten() 
            for k in keys
        }      
        return res_final

        
    def run_SP_model(self):
        self.load_data()
        self.SP_model = gp.Model('SP_model')  
        self.SP_result = {}
        self.SP_model, self.SP_result, self.obj = self.model(self.SP_model, self.SP_result, method='SP', S=self.S)

        self.SP_model.setObjective(self.obj, sense = GRB.MINIMIZE)
        self.SP_model.optimize()   ### 开始问题求解

        self.SP_result['ObjVal'] = self.SP_model.ObjVal  ### 获取求解目标函数值

        if  self.SP_model.status == gp.GRB.OPTIMAL:
            print("HEMS operation: optimal solution found ! \n")
            print(f"Optimal objective value: {round(self.SP_model.ObjVal, 2)} \n")

            result = self.save_results(self.SP_model, self.SP_result, Saved = True,  file_name='HEMS_SP_result.npz')

        elif self.SP_model.status == gp.GRB.INFEASIBLE:
            print("SP_model is infeasible!\n")

        elif self.SP_model.status == gp.GRB.UNBOUNDED:
            print("SP_model is unbounded!\n")

        else:
            print(f"Optimization ended with status: {self.SP_model.status}\n" )
        
        self.time_sp = self.SP_model.Runtime
        return self.SP_model.ObjVal

    def run_Opt_model(self, selected_day, program):
        self.load_data()
        self.Opt_model = gp.Model('Opt_model')  # 定义优化求解器
        self.Opt_result = {}                    # 存储优化结果

        self.Opt_model, self.Opt_result, self.obj = self.model(self.Opt_model, self.Opt_result, method='OPT', S=1, selected_day=selected_day)

        if program == "SP":
            result = np.load('result/HEMS_SP_result.npz')
            P_g_buy_cla   = result["P_g_buy"]
            P_g_sell_cla  = result["P_g_sell"]
            m_B_cla       = result["m_B"]
            
            self.Opt_model.addConstrs(self.P_g_buy[t]  == P_g_buy_cla[t]  for t in range(self.T) )
            self.Opt_model.addConstrs(self.P_g_sell[t] == P_g_sell_cla[t] for t in range(self.T) )
            self.Opt_model.addConstrs(self.m_B[t]      == m_B_cla[t]      for t in range(self.T) )

        elif program == 'PRED':
            result = np.load('result/HEMS_PRED_result.npz')
            P_g_buy_pred   = result["P_g_buy"]
            P_g_sell_pred  = result["P_g_sell"]
            m_B_pred       = result["m_B"]
            
            self.Opt_model.addConstrs(self.P_g_buy[t]  == P_g_buy_pred[t]  for t in range(self.T) )
            self.Opt_model.addConstrs(self.P_g_sell[t] == P_g_sell_pred[t] for t in range(self.T) )
            self.Opt_model.addConstrs(self.m_B[t]      == m_B_pred[t]      for t in range(self.T) )

        
        self.Opt_model.setObjective(self.obj, sense = GRB.MINIMIZE)
        self.Opt_model.setParam('OutputFlag', 0)
        self.Opt_model.optimize()   ### 开始问题求解

        self.time_opt = self.Opt_model.Runtime

        if  self.Opt_model.status == gp.GRB.OPTIMAL:
            self.Opt_result['ObjVal'] = self.Opt_model.ObjVal  ### 获取求解目标函数值
            print("Optimal HEMS operation found ! \n")
            print( f"Optimal HEMS operation cost: {round(self.Opt_model.ObjVal, 2)}\n" )
            #TODO 提取结果中的状态值，找到最后一步的状态
            result = self.save_results(self.Opt_model, self.Opt_result, Saved = False, file_name=f'HEMS_{program}_result_Day{selected_day}.npz')
            #violation = np.sum( self.lambda_ED * result["Q_ED_pos"] + self.lambda_HD * result["Q_HD_pos"] + self.lambda_CD * result["Q_CD_pos"] )  ###计算约束违反产生的惩罚
            violation = np.sum(  result["Q_ED_pos"] +  result["Q_HD_pos"] +  result["Q_CD_pos"] )  #计算约束违反总和
            cost      = np.sum( self.lambda_b[selected_day] * result["P_g_buy"] - self.lambda_s[selected_day] * result["P_g_sell"] + self.lambda_B[selected_day] * result["m_B"] )  #计算总的买电费用（不计约束违反惩罚）

        elif self.Opt_model.status == gp.GRB.INFEASIBLE:
            print("HEMS Opt model is infeasible!\n")

        elif self.Opt_model.status == gp.GRB.UNBOUNDED:
            print("HEMS Opt model is unbounded!\n")

        else:
            print( f"Optimization ended with status: {self.Opt_model.status}\n")
        
        return  -round(self.Opt_model.ObjVal, 2), round(cost, 2), round(violation, 2), result ###保留2位小数
        
    
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



  
        

        


