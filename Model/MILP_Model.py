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

from Model.handler import prediction_train, get_results, test

class HIES_MILP:
    def __init__(self, penalty_factor=10) -> None:
        self.S =  30

        self.env_options = HIES_Options()

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



    def variables_def(self, S, model, result):
        ### 定义决策变量
        self.P_g_buy    = model.addVars(self.T,  lb = 0,  ub = GRB.INFINITY,  name="P_g_buy",  vtype=GRB.CONTINUOUS )  # 与电网买卖电量[单位：kW]
        self.P_g_sell   = model.addVars(self.T,  lb = 0,  ub = GRB.INFINITY,  name="P_g_sell",  vtype=GRB.CONTINUOUS )  # 与电网买卖电量[单位：kW]

        self.m_B     = model.addVars(self.T,  lb = 0,  ub = GRB.INFINITY,  name="m_B",  vtype=GRB.CONTINUOUS )              # 氢市场买氢量[单位：kg]

        self.u_ely    = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="u_ely",  vtype=GRB.CONTINUOUS   )           # 电解池输入功率[单位：kW]
        self.u_fc    = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="u_fc",  vtype=GRB.CONTINUOUS   )           # 燃料电池输出功率[单位：kW]
        self.v_ely   = model.addVars(S, self.T,  lb = 0,  ub = GRB.INFINITY,  name="v_ely",  vtype=GRB.CONTINUOUS   )          # 存储氢气速率

        
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

        return model, result





    def model(self, model, 
              result, 
              params=None, 
              method='PRED', 
              S=1, 
              selected_day=None, 
              T = 24, 
              pred_res = None,
              forecast_data=None
            ):
        if method == 'PRED':
            # 更改预测时间段
            time_start = params['time_start']
            time_end = params['time_end']
            self.T = time_end - time_start

            # Extract forecasting data
            self.lambda_b = forecast_data[:, 0]
            self.solar_radiation = forecast_data[:, 1]
            self.Q_ED = forecast_data[:, 2]
            self.Q_HD = forecast_data[:, 3]
            self.Q_CD = forecast_data[:, 4]
            self.T_a  = forecast_data[:, 5]

            self.P_solar_gen  = self.solar_radiation * self.env_options.S_solar_gen * self.env_options.eta_solar_gen /1000
            
            # Penalty Factor
            self.lambda_ED  = self.penalty_factor * np.mean(self.lambda_b, axis = 0)     # 电需求未满足惩罚因子
            self.lambda_CD  = self.penalty_factor * np.mean(self.lambda_b, axis = 0)     # 热水需求未满足惩罚因子
            self.lambda_HD  = self.penalty_factor * np.mean(self.lambda_b, axis = 0)     # 冷水需求未满足惩罚因子

        else:
            self.T = T

        model, result = self.variables_def(S, model, result)

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
                #TODO u_ely是gurobi变量(var)，无法直接用来比较
                model.addConstr((u_ely > self.env_options.P_ely_low) & (u_ely < self.env_options.P_nom_ely),
                    v_ely == self.env_options.z1 * self.T_ely[s, t] + self.env_options.z0 + self.env_options.z_low * (u_ely - self.env_options.P_nom_ely)
                )

                model.addConstr(
                    (u_ely > self.env_options.P_nom_ely) & (u_ely <= self.env_options.P_ely_high),
                    v_ely == (self.env_options.z1 * (self.T_ely[selected_day, t] if selected_day else self.T_ely[s, t]) 
                            + self.env_options.z0 + self.env_options.z_high * (u_ely - self.env_options.P_nom_ely))
                )

                model.addConstr(
                    (u_ely <= self.env_options.P_ely_low) | (u_ely > self.env_options.P_ely_high),
                    v_ely == 0
                )


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
        decisions = {'P_g_buy': self.P_g_buy, 'P_g_sell': self.P_g_sell, 'm_B': self.m_B}
        return model, result, obj, decisions
