import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


class HIES_Options:
    def __init__(self):

        self.T     = 24    
        self.DELTA =  1    # 时间间隔     



        # 储能设备参数 (BSS)
        self.eta_bssc    = 0.95    # 储能充电效率系数
        self.eta_bssd    = 0.95    # 储能放电效率系数
        self.p_bssc_max  = 40      # 储能最大充电功率（单位：kW）
        self.p_bssd_max  = 40      # 储能最大放电功率（单位：kW）
        self.S_bss_max   = 100     # 储能最大能量水平（单位: kWh）
        self.S_bss_min   = 0       # 储能最小能量水平


        # 电解槽设备参数（EL）
        self.eta_EL = 0.7                               # 电解槽产生氢气的效率
        self.LHV_H2 = 33.2                              # 氢气低热值low heat value (单位：kWh/kg)
        self.k_EL   = self.eta_EL / self.LHV_H2         # 电解槽电转氢效率（单位：kg/kW），0.0211/0.022
        self.eta_EL_rec = 0.8                           # 电解槽热回收效率


        # New
        # electrolyzer
        self.P_ely_low = 15
        self.P_ely_high = 200
        self.q0 = 9.589     # kW
        self.q1 = 1.069
        self.P_nom_ely = 100   # kW
        self.z1 = 1.618 * 10**(-5)
        self.z0 = 1.490 * 10**(-2)
        self.z_low = 1.530 * 10**(-4)
        self.z_high = 1.195 * 10**(-4)
        self.j1 = 0.919
        self.j2 = 7.572 * 10**(-3)
        self.j0 = 3.958
        self.T_ely_high = 70

        self.T_ely_init = 20
        
        self.P_nom_ely_1 = 25  # kW
        self.h1 = 2.565
        self.h0 = 271.7 # A
        self.h_low = 11.11
        self.h_high = 8.276
        self.i_nom_ely = 300 # A
        self.C_high = 75 # Ah

        self.C_init = 0
        
        # storage tank and gas cleaner
        self.alpha = 0.697
        self.b0 = 11.5 * 10**5
        self.b1 = 4.16 * 10**3
        self.V_tank = 40 # m^3
        self.rho = 8.99 * 10**(-2)  
        self.g0 = 0.94
        self.g1 = 5.91 * 10**(-2)
        self.p_low = 12 # bar = 10^6 Pa
        self.p_high = 45 # bar = 10^6 Pa
        self.T_tank_max = self.T_ely_high
        self.T_tank_init = 20

        self.T_a = pd.read_csv('datasets/data/T_a.csv', index_col=0).to_numpy()


        # Fuel cell
        self.u_low = 30  # kW
        self.u_high = 65 # kW
        self.d = 0.96
        self.c = 0.21
        self.s1 = 2.56
        self.s2 = 3.31
        self.u_bp_fc = 47.97
        self.i_bp_fc = 123.05  # A
        


        
























        
        # 氢压缩机设备参数（CO）
        self.k_CO = 3             # 压缩机耗电系数（单位：kW/kg）

        # 燃料电池设备参数（FC）
        self.eta_FC = 0.3                               # 燃料电池氢气发电效率
        self.k_FC   = self.eta_FC * self.LHV_H2         # 燃料电池发电系数（单位：kW/kg）      
        self.eta_FC_rec = 0.8                           # 燃料电池热回收系数
     

        # 储氢罐设备参数 (HSS)
        self.S_hss_max  = 200      #储氢罐最大储氢量（单位：kg）
        self.S_hss_min  = 0        #储氢罐最小储氢量（单位：kg）
        self.m_hssc_max = 50       #储氢罐最大输入量（单位：kg）
        self.m_hssd_max = 50       #储氢罐最大输出量（单位：kg）
        

        # 储热水罐设备参数(TES)
        self.eta_tesc   = 0.9      # 储热水罐输入效率
        self.eta_tesd   = 0.9      # 储热水罐输出效率
        self.g_tesc_max = 40       # 储热水罐最大输入量（单位：kg）
        self.g_tesd_max = 40       # 储热水罐最大输出量（单位：kg）
        self.S_tes_max  = 100      # 储热水罐最大储热水量（单位：kg）
        self.S_tes_min  = 0        # 储热水罐最小储热水量（单位：kg）

        # 吸收式制冷机设备参数（AC）
        self.g_AC_max = 200        # 吸收式制冷机最大输入热量（单位：kW）
        self.eta_AC   = 0.94       # 吸收式制冷机的回收效率（取值范围0~1）

        # 储冷水罐设备参数 (CSS)
        self.eta_cssc   = 0.9 
        self.eta_cssd   = 0.9
        self.q_cssc_max = 40     # 储冷水罐最大输入量（单位：kg）
        self.q_cssd_max = 40     # 储冷水罐最大输出量（单位：kg）
        self.S_css_max  = 200    # 储冷水罐最大储冷水量（单位：kg）
        self.S_css_min  = 0      # 储冷水罐最小储冷水量（单位：kg）

        # 光伏发电和光伏集热器设备参数
        self.S_solar_gen =  3000  # 光伏电站太阳板面积（单位：m2）
        self.eta_solar_gen = 0.2  # 太阳板转化效率（取值范围：0~1）


        self.S_solar_heat = 400      # 光伏电站太阳板面积（单位：m2）
        self.eta_solar_heat = 0.762   # 太阳能集热器转化效率（取值范围：0~1）

      

        # 光伏发电功率数据
        self.solar_radiation = pd.read_csv("datasets/data/solar_radiation.csv", index_col=0).to_numpy()              ### 太阳辐照强度（单位：W/m2）
        self.P_solar_gen = self.solar_radiation * self.S_solar_gen * self.eta_solar_gen /1000                           ### 光伏发电功率（单位：kW）

        # 光伏集热器制热数据
        self.P_solar_heat = self.solar_radiation * self.S_solar_heat * self.eta_solar_heat/1000       ### 光伏集热器制热（单位：kW）

        # 电力市场买电、卖电价格
        self.lambda_b = pd.read_csv("datasets/data/electricity_price.csv", index_col=0).to_numpy()       ### 从电网买电价格（单位：元/kWh）
        self.lambda_s = np.zeros(self.lambda_b.shape)     
        self.lambda_B = 0.1 * np.ones(self.lambda_b.shape)     #买氢价格（单位：元/kg）                                                                          ### 向电网卖电价格（单位：元/kWh）
  
        # 电、热水、冷水需求量 
        self.Q_ED = pd.read_csv("datasets/data/electricity_demand.csv", index_col = 0 ).to_numpy()   ### 电需求   (功率：200kW)
        self.Q_HD = pd.read_csv("datasets/data/heating_demand.csv",     index_col = 0 ).to_numpy()    ### 热水需求（功率：100kW）
        self.Q_CD = pd.read_csv("datasets/data/cooling_demand.csv",     index_col = 0 ).to_numpy()/3   ### 冷水需求（功率：100kW）

        
        # 储能设备状态初始化
        self.S_bss_init = 0.1 * self.S_bss_max
        self.S_hss_init = 0.1 * self.S_hss_max
        self.S_tes_init = 0.1 * self.S_tes_max
        self.S_css_init = 0.1 * self.S_css_max

        # 能量流约束
        self.P_g_max   = 600    # 电力市场买卖电功率上限（单位：kW）
        self.P_EL_max  = 500    # 电解槽最大输入功率（单位：kW）
        self.P_FC_max  = 500    # 燃料电池最大发电量（单位：kW）(计算值：91)
        self.m_B_max   = 20    # 氢市场最大买氢量（单位:kg）

        self.MAX_DAY  = min( self.lambda_b.shape[0], self.lambda_s.shape[0], self.P_solar_gen.shape[0], self.Q_ED.shape[0], self.Q_HD.shape[0], self.Q_CD.shape[0] )

        ###设置约束违反惩罚因子
        self.lambda_ED  = np.mean(self.lambda_b, axis = 0)     # 电需求未满足惩罚因子
        self.lambda_CD  = np.mean(self.lambda_b, axis = 0)     # 热水需求未满足惩罚因子
        self.lambda_HD  = np.mean(self.lambda_b, axis = 0)     # 冷水需求未满足惩罚因子

        self.penalty1 = 1   # 买点成本
        self.penalty2 = 10  # T_ely
        self.penalty3 = 10  # C(t)
        self.penalty4 = 10  # p(t)
        self.penalty5 = 10  # balance

        

    
