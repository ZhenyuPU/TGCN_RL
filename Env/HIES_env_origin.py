import pandas as pd
import random
import numpy as np
import os
from Utils.HIES_options import HIES_Options

random.seed(0)
np.random.seed(0)

"""
    原始神经网络版本，且没有更改EL和FC模型
"""

### 构建hydrogen-based multi-energy system (HEMS)
class HIES_Env:
    """
        states:
            electricity purchase price (backward K steps), electricity selling price (backward K steps),  hydorgen purchase price (backward K steps)
            battery_level, hydrogen_level, hot_water_level, chilled_water_level,
            P_solar_gen (backward K steps)
            Q_ED, Q_HD, Q_CD (backward K steps)  
            t  (current time)  
        actions:
            P_g, P_EL, P_bssc, P_bssd, m_B, P_FC, g_AC, g_tesc, g_tesd, q_cssc, q_cssd
    """
    def __init__(self, HIES_Options, penalty_factor=10):

        self.env_options   = HIES_Options
        self.start         = 0
        
        self.day           = 0      # 从第1天的数据开始训练
        self.time_step     = 0      # 从0时刻开始训练
        self.K             = 0      # 考虑历史K步的信息

        self.T             = 23     # 24小时
        self.MAX_DAY       = self.env_options.MAX_DAY

        self.state_dim      = 12     # 总共有12个状态分量
        self.action_dim     = 7      # 总共有8个行动分量

        self.DELTA          = 1      # 时间间隔1小时
        
        self.P_solar_gen   =  self.env_options.P_solar_gen   # 光伏发电数据
        self.P_solar_heat  =  self.env_options.P_solar_heat  # 光伏集热器数据

        self.lambda_b      =  self.env_options.lambda_b      # 电力市场买电价格
        self.lambda_s      =  self.env_options.lambda_s      # 电力市场卖电价格
        self.lambda_B      =  self.env_options.lambda_B      # 氢市场买氢价格

        self.lambda_ED    =   self.env_options.lambda_ED * penalty_factor      # 电需求未满足惩罚因子
        self.lambda_HD    =   self.env_options.lambda_HD * penalty_factor      # 热水需求未满足的惩罚因子
        self.lambda_CD    =   self.env_options.lambda_CD * penalty_factor      # 冷水需求未满足的惩罚因子
        
        self.Q_ED         =   self.env_options.Q_ED             # 用户电需求
        self.Q_HD         =   self.env_options.Q_HD             # 用户热水需求
        self.Q_CD         =   self.env_options.Q_CD             # 用户冷水需求

        self.state = self.get_initial_state()   ### 获取当日的初始状态 （Env存储的是原始状态，未归一化）

    def get_info(self):
        print('day = {}, time_step = {}'.format(self.day, self.time_step))
        return self.day, self.time_step   # 返回当前日期和当前时刻
    

    def reset(self):   # 初始化日期为第1天，初始为时刻为0时刻，初始化系统状态为第1天、0时刻的状态
        self.day       =  random.randint(0, self.MAX_DAY - 2)  ### 随机选一天数据
        self.time_step = 0 
        self.state     = self.get_initial_state()
        #print('reset')
        return self.state_normalization(self.state)
        #return self.state  #返回归一化后的状态
    

    def get_initial_state(self):    # 获取每一天0时刻的状态值

        state     =  np.zeros( self.state_dim )

        state[0]  =  self.lambda_b[self.day][0]    # 电力市场买电价格
        state[1]  =  self.lambda_s[self.day][0]    # 电力市场卖电价格
        state[2]  =  self.lambda_B[self.day][0]    # 氢市场买氢价格

        state[3]  = self.env_options.S_bss_init     # 初始储能能量状态
        state[4]  = self.env_options.S_hss_init     # 初始储氢罐状态
        state[5]  = self.env_options.S_tes_init     # 初始储热水罐状态
        state[6]  = self.env_options.S_css_init     # 初始储冷水罐状态

        state[7]  = self.P_solar_gen[self.day][0]   # 光伏发电

        state[8]  = self.Q_ED[self.day][0]          # 电需求
        state[9]  = self.Q_HD[self.day][0]          # 热水需求
        state[10] = self.Q_CD[self.day][0]          # 冷水需求

        state[11] = 0                               # 初始时刻为0
        return state                                # 返回初始状态（未归一化）
    
    def step(self, action):                         # 执行一步决策，更新系统状态
        """
        input: action: a_g, a_EL, a_FC, a_B, a_AC, a_bss, a_tes, a_css
        output: next_state, reward, done 
        """

        reward, violation, cost = self.get_reward(action)   ### 先获取reward再更新状态

       # 解析当前状态
        current_S_bss = self.state[3]    # 当前储电状态
        current_S_hss = self.state[4]    # 当前储氢量
        current_S_tes = self.state[5]    # 当前储热水量
        current_S_css = self.state[6]    # 当前储冷水量

       # 解析决策（保证设备运行的物理约束）
        decision =  self.get_decision( action )
        P_g      =  decision['P_g'   ] 
        P_EL     =  decision['P_EL'  ] 
        P_FC     =  decision['P_FC'  ] 
        P_CO     =  decision['P_CO'  ]
        m_B      =  decision['m_B'   ] 
        P_bssc   =  decision['P_bssc'] 
        P_bssd   =  decision['P_bssd'] 
        g_FC     =  decision['g_FC'  ]
        g_EL     =  decision['g_EL']
        g_AC     =  decision['g_AC'  ]
        g_tesc   =  decision['g_tesc'] 
        g_tesd   =  decision['g_tesd'] 
        q_cssc   =  decision['q_cssc'] 
        q_cssd   =  decision['q_cssd']

        
        # 更新储能状态
        next_S_bss = current_S_bss  + ( P_bssc * self.env_options.eta_bssc + P_bssd * self.env_options.eta_bssd ) * self.DELTA
        next_S_hss = current_S_hss  + ( m_B    +  self.env_options.k_EL * P_EL - P_FC/self.env_options.k_FC     ) * self.DELTA
        next_S_tes = current_S_tes  + ( g_tesc * self.env_options.eta_tesc + g_tesd/self.env_options.eta_tesd   ) * self.DELTA
        next_S_css = current_S_css  + ( q_cssc * self.env_options.eta_cssc + q_cssd/self.env_options.eta_cssd   ) * self.DELTA

        next_state = np.zeros( self.state_dim )

        if self.time_step < self.T:

            next_state[0]  =  self.lambda_b[self.day][self.time_step + 1 ] 
            next_state[1]  =  self.lambda_s[self.day][self.time_step + 1 ]
            next_state[2]  =  self.lambda_B[self.day][self.time_step + 1 ]

            next_state[3]  =  next_S_bss
            next_state[4]  =  next_S_hss
            next_state[5]  =  next_S_tes
            next_state[6]  =  next_S_css
            
            next_state[7]  = self.P_solar_gen[self.day][self.time_step + 1]

            next_state[8]  = self.Q_ED[self.day][self.time_step + 1 ]
            next_state[9]  = self.Q_HD[self.day][self.time_step + 1 ]
            next_state[10] = self.Q_CD[self.day][self.time_step + 1 ]

            next_state[11] = self.time_step + 1
            self.time_step = self.time_step + 1  # 更新当前时刻
            done = False
        else:
            self.day        = self.day + 1
            self.time_step  = 0
            next_state      = self.get_initial_state()
            done            = True
        
        self.state = next_state ### 更新当前时刻的系统状态
        #return next_state, reward, done
        return self.state_normalization(next_state), reward, done, violation, cost   ### 返回归一化后的状态
    
    
    def state_normalization(self, state):
        """
            states:
                electricity purchase price (backward K steps), electricity selling price (backward K steps),  hydorgen purchase price (backward K steps)
                battery_level, hydrogen_level, hot_water_level, chilled_water_level,
                P_solar_gen (backward K steps)
                Q_ED, Q_HD, Q_CD (backward K steps)  
                t  (current time)  
        """
        Q_max        = max(np.max(self.env_options.P_solar_gen), np.max(self.env_options.Q_ED), np.max(self.env_options.Q_HD), np.max(self.env_options.Q_CD))   # 发电和需求的功率最大值

        new_state = np.zeros(self.state_dim)

        new_state[0] = ( state[0] - 0 )/np.max( self.env_options.lambda_b[self.day] )   # 电力市场买电价格归一化，按天归一化
        new_state[1] = 0 #( state[1] - 0 )/np.max( self.env_options.lambda_s )   # 电力市场卖电价格归一化
        new_state[2] = 0 #( state[2] - 0 )/np.max( self.env_options.lambda_B )   # 电力市场买氢价格归一化
        S_max        = max(self.env_options.S_bss_max, self.env_options.S_hss_max, self.env_options.S_tes_max, self.env_options.S_css_max, Q_max)   # 储能水平最大值
        new_state[3] = ( state[3] - 0 )/S_max              # 储能电能量状态
        new_state[4] = ( state[4] - 0 )/S_max              # 储氢罐储氢量
        new_state[5] = ( state[5] - 0 )/S_max              # 储热水罐储热水量
        new_state[6] = ( state[6] - 0 )/S_max              # 储冷水罐储冷水量


        new_state[7] = ( state[7] - 0 )/S_max          # 光伏发电量
    
        new_state[8]  = (state[8]  - 0 )/S_max         # 电需求
        new_state[9]  = (state[9]  - 0 )/S_max         # 热水需求
        new_state[10] = (state[10] - 0 )/S_max         # 冷水需求
        new_state[11] = (state[11] - 0)/self.T                                   # 当前时刻 （0~23） time_step
    

        return new_state                                                         # 返回归一化后的状态

    
    def get_decision(self, action):
        """
            获取所有的决策变量，同时考虑设备物理约束
        """
        # 解析当前状态
        current_S_bss = self.state[3]    # 当前储电状态
        current_S_hss = self.state[4]    # 当前储氢量
        current_S_tes = self.state[5]    # 当前储热水量
        current_S_css = self.state[6]    # 当前储冷水量
       # 解析决策（保证设备运行的物理约束）
        a_g, a_bss, a_hss, a_B, a_tes, a_AC, a_css = action[0], action[1], action[2], action[3], action[4], action[5], action[6]

        P_g  =  a_g * self.env_options.P_g_max                        # 电力市场买电量（若为负值，表示向电网卖电）

        ### 氢能装置决策
        # 设置为不从氢市场买氢能
        m_B = 0

        P_EL = np.clip( a_hss * self.env_options.P_EL_max, 
                       0,
                       ( self.env_options.S_hss_max - current_S_hss - m_B )/( self.DELTA * self.env_options.k_EL ) )             # 电解槽输入功率
        P_FC = - np.clip( a_hss * self.env_options.P_FC_max,
                       ( self.env_options.S_hss_min - current_S_hss ) * self.env_options.k_FC / self.DELTA,  
                       0)               # 燃料电池发电功率，原本为负数，设置为正数

        P_CO    = self.env_options.k_CO * self.env_options.k_EL * P_EL  # 氢压缩机耗电量
        
        ### BESS
        P_bssc = np.clip( a_bss * self.env_options.p_bssc_max, 0, (self.env_options.S_bss_max - current_S_bss)/( self.env_options.eta_bssc * self.DELTA ) )  ### 储能充电量
        P_bssd = np.clip( a_bss * self.env_options.p_bssd_max, (self.env_options.S_bss_min - current_S_bss) * self.env_options.eta_bssd/self.DELTA,  0 )  ### 储能放电量（负值）

        ### TES
        g_FC   = self.env_options.eta_FC_rec * (1 - self.env_options.eta_FC) * P_FC/self.env_options.eta_FC                       # 燃料电池产热量
        g_EL   = self.env_options.eta_EL_rec * (1 - self.env_options.eta_EL) * P_EL                                               # 电解槽产热量
        g_AC   = np.clip( a_AC * self.env_options.g_AC_max, 0, g_EL + g_FC + self.env_options.P_solar_heat[self.day][self.time_step] )   # 吸收式制冷机消耗热量 不能超过 燃料电池产热 + 集热器吸热 + 电解槽产热

        g_tesc  = np.clip( a_tes * self.env_options.g_tesc_max, 0, (self.env_options.S_tes_max - current_S_tes)/(self.env_options.eta_tesc * self.DELTA)   )     ###热水罐输入量
        g_tesd  = np.clip( a_tes * self.env_options.g_tesd_max,    (self.env_options.S_tes_min - current_S_tes) * self.env_options.eta_tesd/self.DELTA,  0 )     ###热水罐输出量（负值）

        ### CSS
        q_cssc  = np.clip( a_css * self.env_options.q_cssc_max, 0,  (self.env_options.S_css_max - current_S_css)/(self.env_options.eta_cssc * self.DELTA)  )     ### 冷水罐输入量
        q_cssd  = np.clip( a_css * self.env_options.q_cssd_max,  (self.env_options.S_css_min - current_S_css) * self.env_options.eta_cssd/self.DELTA,    0 )     ### 冷水罐输出量（负值）
        
        decision = {}
        decision['P_g'   ]  =  P_g
        decision['P_EL'  ]  =  P_EL
        decision['P_FC'  ]  =  P_FC
        decision['P_CO'  ]  =  P_CO
        decision['m_B'   ]  =  m_B
        decision['P_bssc']  =  P_bssc
        decision['P_bssd']  =  P_bssd
        decision['g_FC'  ]  =  g_FC
        decision['g_AC'  ]  =  g_AC
        decision['g_tesc']  =  g_tesc
        decision['g_tesd']  =  g_tesd
        decision['q_cssc']  =  q_cssc
        decision['q_cssd']  =  q_cssd
        decision['g_EL']    =  g_EL

        return decision
    
    
    def get_reward(self, action):

        P_solar_gen    = self.state[7]      

        # 解析决策（保证设备运行的物理约束）
        decision =  self.get_decision( action )
        P_g      =  decision['P_g'   ] 
        P_EL     =  decision['P_EL'  ] 
        P_FC     =  decision['P_FC'  ] 
        P_CO     =  decision['P_CO'  ]
        m_B      =  decision['m_B'   ] 
        P_bssc   =  decision['P_bssc'] 
        P_bssd   =  decision['P_bssd'] 
        g_FC     =  decision['g_FC'  ]
        g_EL     =  decision['g_EL']
        g_AC     =  decision['g_AC'  ]
        g_tesc   =  decision['g_tesc'] 
        g_tesd   =  decision['g_tesd'] 
        q_cssc   =  decision['q_cssc'] 
        q_cssd   =  decision['q_cssd']
 
        # c1为买电费用、c2为买氢费用、c3为需求未满足的惩罚
        c1  =  ( self.lambda_b[self.day][self.time_step] - self.lambda_s[self.day][self.time_step] )/2 * np.abs(P_g) + ( self.lambda_b[self.day][self.time_step] + self.lambda_s[self.day][self.time_step] )/2 * P_g

        c2  =    self.lambda_B[self.day][self.time_step] * m_B

        c3  =  (      self.lambda_ED[self.time_step] * max( self.Q_ED[self.day][self.time_step] + P_EL + P_CO + P_bssc + P_bssd - P_g - P_solar_gen - P_FC, 0 )
                   +  self.lambda_CD[self.time_step] * max( self.Q_CD[self.day][self.time_step] + q_cssc + q_cssd - g_AC * self.env_options.eta_AC, 0  )
                   +  self.lambda_HD[self.time_step] * max( self.Q_HD[self.day][self.time_step] + g_tesc + g_tesd - g_FC - g_EL - self.env_options.P_solar_heat[self.day][self.time_step] + g_AC, 0 ) 
               ) 

        violation = (     max( self.Q_ED[self.day][self.time_step] + P_EL + P_CO + P_bssc + P_bssd - P_g - P_solar_gen - P_FC, 0 )
                        + max( self.Q_CD[self.day][self.time_step] + q_cssc + q_cssd - g_AC * self.env_options.eta_AC, 0  )
                        + max( self.Q_HD[self.day][self.time_step] + g_tesc + g_tesd - g_FC - g_EL - self.env_options.P_solar_heat[self.day][self.time_step] + g_AC, 0 ) 
                    )
        reward = -(c1 * self.DELTA + c2 * self.DELTA + c3)
        cost = c1 + c2
        return reward, violation, cost


### 测试环境
if __name__ == '__main__':
   HEMS_Options = HIES_Options()
   HEMS_Env = HIES_Env( HEMS_Options )   ### 创建一个HEMS系统

   action_dim =  8   # 总共包含8个行动分量
   state_dim  =  12  # 总共包含12个状态分量


   def generate_random_action():
        a_g   = random.uniform(-1, 1)
        a_EL  = random.uniform(0,  1)
        a_FC  = random.uniform(0,  1)
        a_B   = random.uniform(0,  1)
        a_AC  = random.uniform(0,  1)
        a_bss = random.uniform(-1, 1)
        a_tes = random.uniform(-1, 1)
        a_css = random.uniform(-1, 1)

        action = np.zeros(action_dim)
        action[0] = a_g
        action[1] = a_EL
        action[2] = a_FC
        action[3] = a_B
        action[4] = a_AC
        action[5] = a_bss
        action[6] = a_tes
        action[7] = a_css
        return action

   MAX_EPOCH = 1

   for epoch in range(MAX_EPOCH):

        done = False 

        trace = {} ### store the trajectories
        trace['state' ]   = []
        trace['action']   = []
        trace['decision'] = []
          

        while not done: 
            state  = HEMS_Env.state                        # 获取当前状态
            action  = generate_random_action()             # 随机产生一个行动
            decision = HEMS_Env.get_decision(action)

            next_state, reward, done = HEMS_Env.step(action) 

            trace['state'].append(state)     #存储当前状态
            trace['action'].append(action)   #存储当前行动
            trace['decision'].append(decision)
            

            day, time_step = HEMS_Env.get_info()
            

   




    
    
    

    
