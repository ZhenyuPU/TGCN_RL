"""
    HIES Environment Settings
    States:
        0 electricity purchase price[a1, a2, ..., a24]   
        1 PV power[a1, a2, ..., a24]
        2 electricity demand[a1, a2, ..., a24]
        3 heating demand[a1, a2, ..., a24]
        4 cooling demand[a1, a2, ..., a24]

        5 battery storage level
        6 hydrogen tank level m_t
        7 heating water tank level
        8 cooling water tank level

        9 T_ely
        10 C_t
        11 T_tank
        12 T_a: environment temperature

        13 t
        
    Actions:
        0 a_g
        1 a_bss
        2 a_tes
        3 a_AC
        4 a_css
        5 a_ely
        6 a_fc
    Reward:

    State Transitions:

    Elctrolyzers:
        4 stacks
    Fuel cell:
        2 stacks
"""



#TODO 将env改造为gym的子类

#TODO 改造网络


from Utils.HIES_options import HIES_Options
import numpy as np
import random
import matplotlib.pyplot as plt

"""
    原始神经网络版本，但是已经更改EL和FC模型
"""


class HIES_Env:
    def __init__(self):
        self.env_options = HIES_Options()
        self.MAX_DAY = self.env_options.MAX_DAY # 数据集最大天数
        self.days = 0       # 选择episode day
        self.T = 23         # 一个episode 24h
        self.time_step = 0  # 每一个episode从0开始
        self.K = 0          # 考虑历史K步信息
        self.DELTA = 1      # time interval

        self.state_dim = 14 # state dimension
        self.action_dim = 7 # action dimension

    def reset(self):
        """
            重新选择某一天，每个episode结束之后
        """
        self.day = random.randint(0, self.MAX_DAY)
        self.time_step = 0
        self.states = self.get_initial_states()
        return  self.state_normalization(self.states)
    
    def state_normalization(self, states):
        """
            状态归一化
        """
        states_norm = np.zeros(self.state_dim)

        Q_max        = max(np.max(self.env_options.P_solar_gen), 
                           np.max(self.env_options.Q_ED), 
                           np.max(self.env_options.Q_HD), 
                           np.max(self.env_options.Q_CD), 
                           np.max(self.env_options.T_a))   # 发电和需求的功率最大值
        S_max        = max(self.env_options.S_bss_max, 
                           self.env_options.S_hss_max, self.env_options.S_tes_max, 
                           self.env_options.S_css_max, 
                           Q_max, 
                           self.env_options.T_ely_high, 
                           self.env_options.C_high, 
                           self.env_options.T_tank_max)   # 储能水平最大值

        states_norm[0] = ( states[0] - 0)/np.max( self.env_options.lambda_b[self.day] )   # 电力市场买电价格归一化，按天归一化
        states_norm[1] = (states[1] - 0)/S_max          # 光伏发电量
        states_norm[2] = (states[2] - 0)/S_max         # 电需求
        states_norm[3] = (states[3] - 0)/S_max         # 热水需求
        states_norm[4] = (states[4] - 0)/S_max         # 冷水需求

        states_norm[5] = (states[5] - 0 )/S_max              # 储能电能量状态
        states_norm[6] = (states[6] - 0 )/S_max              # 储氢罐储氢量
        states_norm[7] = (states[7] - 0 )/S_max              # 储热水罐储热水量
        states_norm[8] = (states[8] - 0 )/S_max              # 储冷水罐储冷水量

        states_norm[9]  = (states[9]  - 0 )/S_max            # T_ely
        states_norm[10] = (states[10] - 0 )/S_max            # C(t)
        states_norm[11] = (states[11] - 0 )/S_max            # T_tank
        states_norm[12] = (states[12] - 0 )/S_max            # T_a

        states_norm[13] = (states[13] - 0)/self.T              # 当前时刻 （0~23）
        return states_norm
    
    def get_initial_states(self):
        states = np.zeros(self.state_dim)
        self.time_step = 0
        states[0] = self.env_options.lambda_b[self.day][self.time_step]
        states[1] = self.env_options.P_solar_gen[self.day][self.time_step]
        states[2] = self.env_options.Q_ED[self.day][self.time_step]
        states[3] = self.env_options.Q_HD[self.day][self.time_step]
        states[4] = self.env_options.Q_CD[self.day][self.time_step]

        states[5] = self.env_options.S_bss_init
        states[6] = self.env_options.S_hss_init
        states[7] = self.env_options.S_tes_init
        states[8] = self.env_options.S_css_init

        states[9]  = self.env_options.T_ely_init                               # T_ely
        states[10] = self.env_options.C_init                                   # C(t)
        states[11] = self.env_options.T_tank_init                              # T_tank
        states[12] = self.env_options.T_a[self.day][self.time_step]            # T_a

        states[13] = self.time_step
        return states
    

    def get_decisions(self, actions):
        """获取决策变量的值

        Args:
            actions (_type_): _description_
        """
        # 获取当前actions
        # 不考虑买氢
        a_g, a_bss, a_tes, a_AC, a_css, a_ely, a_fc = actions
        # 解析当前状态
        current_S_bss = self.states[5]    # 当前储电状态
        current_S_hss = self.states[6]    # 当前储氢量
        current_S_tes = self.states[7]    # 当前储热水量
        current_S_css = self.states[8]    # 当前储冷水量
        T_ely = self.states[9]               #TODO 温度
        C_t   = self.states[10]              #TODO C(t)
        T_tank   = self.states[11]              #TODO T_tank(t)

        P_g = a_g * self.env_options.P_g_max        # 购电功率

        # BESS
        # 电池充电
        P_bssc = np.clip(a_bss * self.env_options.p_bssc_max, 0, (self.env_options.S_bss_max - current_S_bss)/( self.env_options.eta_bssc * self.DELTA ) )
        # 电池放电（负值）
        P_bssd = np.clip( a_bss * self.env_options.p_bssd_max, (self.env_options.S_bss_min - current_S_bss) * self.env_options.eta_bssd/self.DELTA,  0 )

        # TES
        # 热水罐输入量
        g_tesc  = np.clip( a_tes * self.env_options.g_tesc_max, 0, (self.env_options.S_tes_max - current_S_tes)/(self.env_options.eta_tesc * self.DELTA)   )
        # 热水罐输出量（负值）     
        g_tesd  = np.clip( a_tes * self.env_options.g_tesd_max,    (self.env_options.S_tes_min - current_S_tes) * self.env_options.eta_tesd/self.DELTA,  0 )     
        # CSS
        # 冷水罐输入量
        q_cssc  = np.clip( a_css * self.env_options.q_cssc_max, 0,  (self.env_options.S_css_max - current_S_css)/(self.env_options.eta_cssc * self.DELTA)  )  
        # 冷水罐输出量（负值）
        q_cssd  = np.clip( a_css * self.env_options.q_cssd_max,  (self.env_options.S_css_min - current_S_css) * self.env_options.eta_cssd/self.DELTA,    0 )  

        # AC吸收式制冷器

        # HESS
        m_B =  0        # 买氢量为0
        # Electrolyzer
        P_ely = self.env_options.P_ely_low + (self.env_options.P_ely_high - self.env_options.P_ely_low) * a_ely     # AC power
        u_ely = (P_ely - self.env_options.q0) / self.env_options.q1   # DC Power
        # volume rate of hydrogen
        if u_ely > self.env_options.P_ely_low and u_ely < self.env_options.P_nom_ely:
            v_ely = self.env_options.z1 * T_ely + self.env_options.z0 + self.env_options.z_low * (u_ely - self.env_options.P_nom_ely)
        elif u_ely > self.env_options.P_nom_ely and u_ely <= self.env_options.P_ely_high:
            v_ely = self.env_options.z1 * T_ely + self.env_options.z0 + self.env_options.z_high * (u_ely - self.env_options.P_nom_ely)
        else:
            v_ely = 0

        # Overload counter model
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
        
        p_t = (self.env_options.b0 + self.env_options.b1 * T_tank) * (current_S_hss / self.env_options.V_tank) / (10**6)
        
        # Fuel cell
        u_fc = self.env_options.u_low + (self.env_options.u_high - self.env_options.u_low) * a_fc  # DC power
        P_fc = self.env_options.d * u_fc    # AC power
        # Current of FC
        if self.env_options.u_low < u_fc <= self.env_options.u_bp_fc:
            i_fc = self.env_options.s1 * u_fc
        elif self.env_options.u_bp_fc < u_fc <= self.env_options.u_high:
            i_fc = self.env_options.s2 * (u_fc - self.env_options.u_bp_fc) + self.env_options.i_bp_fc
        else:
            i_fc = 0
        v_fc = self.env_options.c * i_fc

        g_FC = self.env_options.eta_FC_rec * (1 - self.env_options.eta_FC) * u_fc / self.env_options.eta_FC                       # 燃料电池产热量
        g_EL   = self.env_options.eta_EL_rec * (1 - self.env_options.eta_EL) * u_ely                                               # 电解槽产热量

        g_AC   = np.clip( a_AC * self.env_options.g_AC_max, 0, g_EL + g_FC + self.env_options.P_solar_heat[self.day][self.time_step] )   # 吸收式制冷机消耗热量 不能超过 燃料电池产热 + 集热器吸热 + 电解槽产热
        
        # 保存决策变量
        decisions = {}
        decisions['u_ely']  = u_ely
        decisions['P_ely']  = P_ely
        decisions['v_ely']  = v_ely

        decisions['P_ely_1']  = P_ely_1
        decisions['i_ely']  = i_ely

        decisions['v_cdg']  = v_cdg
        decisions['p_t']  = p_t

        decisions['u_fc']  = u_fc
        decisions['P_fc']  = P_fc
        decisions['i_fc']  = i_fc
        decisions['v_fc']  = v_fc

        decisions['P_g']  = P_g
        decisions['P_bssc']  =  P_bssc
        decisions['P_bssd']  =  P_bssd
        decisions['g_tesc']  =  g_tesc
        decisions['g_tesd']  =  g_tesd
        decisions['q_cssc']  =  q_cssc
        decisions['q_cssd']  =  q_cssd

        decisions['g_FC']  =  g_FC
        decisions['g_EL']  =  g_EL
        decisions['g_AC']  =  g_AC

        return decisions
    
    def step(self, actions):
        # 获取reward
        reward, cost, violation = self.get_reward(actions)

        # 解析当前状态
        current_S_bss = self.states[5]    # 当前储电状态
        current_S_hss = self.states[6]    # 当前储氢量
        current_S_tes = self.states[7]    # 当前储热水量
        current_S_css = self.states[8]    # 当前储冷水量
        T_ely = self.states[9]               #TODO 温度
        C_t   = self.states[10]              #TODO C(t)
        T_tank   = self.states[11]              #TODO T_tank(t)

        # 解析决策
        decisions = self.get_decisions(actions)
        P_g      =  decisions['P_g'   ]   
        P_bssc   =  decisions['P_bssc'] 
        P_bssd   =  decisions['P_bssd'] 
        g_FC     =  decisions['g_FC'  ]
        g_EL     =  decisions['g_EL']
        g_AC     =  decisions['g_AC'  ]
        g_tesc   =  decisions['g_tesc'] 
        g_tesd   =  decisions['g_tesd'] 
        q_cssc   =  decisions['q_cssc'] 
        q_cssd   =  decisions['q_cssd']
        u_ely    =  decisions['u_ely']
        i_ely    =  decisions['i_ely']
        v_cdg    =  decisions['v_cdg']
        v_fc     =  decisions['v_fc']



        # 更新储能状态
        next_S_bss = current_S_bss  + ( P_bssc * self.env_options.eta_bssc + P_bssd * self.env_options.eta_bssd ) * self.DELTA
        next_S_tes = current_S_tes  + ( g_tesc * self.env_options.eta_tesc + g_tesd/self.env_options.eta_tesd   ) * self.DELTA
        next_S_css = current_S_css  + ( q_cssc * self.env_options.eta_cssc + q_cssd/self.env_options.eta_cssd   ) * self.DELTA

        # Temperature of EL stacks
        T_ely_next = T_ely * self.env_options.j1 + self.env_options.j2 * u_ely + self.env_options.j0
        # Current counter dynamics
        C_next = max(0, C_t + self.DELTA * (i_ely - self.env_options.i_nom_ely))
        # Storage tank
        m_next = current_S_hss + self.env_options.rho * (v_cdg - v_fc)
        #TODO Tank temperature 
        T_tank_next = T_tank * self.env_options.g0 + self.env_options.g1 * self.env_options.T_a[self.day][self.time_step] 

        next_state = np.zeros( self.state_dim )

        if self.time_step < self.T:

            next_state[0]  =  self.env_options.lambda_b[self.day][self.time_step + 1 ] 
            next_state[1]  = self.env_options.P_solar_gen[self.day][self.time_step + 1]

            next_state[2]  = self.env_options.Q_ED[self.day][self.time_step + 1 ]
            next_state[3]  = self.env_options.Q_HD[self.day][self.time_step + 1 ]
            next_state[4] = self.env_options.Q_CD[self.day][self.time_step + 1 ]

            next_state[5]  =  next_S_bss
            next_state[6]  =  m_next
            next_state[7]  =  next_S_tes
            next_state[8]  =  next_S_css

            next_state[9]  =  T_ely
            next_state[10] =  C_t
            next_state[11] =  T_tank_next
            next_state[12] =  T_ely_next

        
            next_state[-1] = self.time_step + 1
            self.time_step = self.time_step + 1  # 更新当前时刻
            done = False
        else:
            self.day        = self.day + 1
            self.time_step  = 0
            next_state      = self.get_initial_states()
            done            = True
        
        self.states = next_state ### 更新当前时刻的系统状态
        #return next_state, reward, done
        return self.state_normalization(next_state), reward, done, violation, cost   ### 返回归一化后的状态
    
    def get_reward(self, actions):
        # 解析当前状态
        current_S_bss = self.states[5]    # 当前储电状态
        current_S_hss = self.states[6]    # 当前储氢量
        current_S_tes = self.states[7]    # 当前储热水量
        current_S_css = self.states[8]    # 当前储冷水量
        P_solar_gen   = self.states[1]    # PV发电功率
        T_ely = self.states[9]               #TODO 温度
        C_t   = self.states[10]              #TODO C(t)
        T_tank   = self.states[11]              #TODO T_tank(t)

        # 解析决策（保证设备运行的物理约束）
        decisions =  self.get_decisions( actions )
        P_g      =  decisions['P_g'   ]   
        P_bssc   =  decisions['P_bssc'] 
        P_bssd   =  decisions['P_bssd'] 
        g_FC     =  decisions['g_FC'  ]
        g_EL     =  decisions['g_EL']
        g_AC     =  decisions['g_AC'  ]
        g_tesc   =  decisions['g_tesc'] 
        g_tesd   =  decisions['g_tesd'] 
        q_cssc   =  decisions['q_cssc'] 
        q_cssd   =  decisions['q_cssd']
        u_ely    =  decisions['u_ely']
        i_ely    =  decisions['i_ely']
        v_cdg    =  decisions['v_cdg']
        v_fc     =  decisions['v_fc']
        p_t      =  decisions['p_t']
        u_fc     =  decisions['u_fc']

        # 买电成本
        C1 = ( self.env_options.lambda_b[self.day][self.time_step] - self.env_options.lambda_s[self.day][self.time_step] )/2 * np.abs(P_g) + ( self.env_options.lambda_b[self.day][self.time_step] + self.env_options.lambda_s[self.day][self.time_step] )/2 * P_g
        # T_ely penalty
        C2 = max(T_ely - self.env_options.T_ely_high, 0)
        # C(t)
        C3 = max(C_t - self.env_options.C_high, 0)
        # p(t)
        C4 = max(p_t - self.env_options.p_high, 0) + max(self.env_options.p_low - p_t, 0)
        #TODO T_tank ???
        # balance violation
        C5 = (     max( self.env_options.Q_ED[self.day][self.time_step] + u_ely + P_bssc + P_bssd - P_g - P_solar_gen - u_fc, 0 )
                + max( self.env_options.Q_CD[self.day][self.time_step] + q_cssc + q_cssd - g_AC * self.env_options.eta_AC, 0  )
                + max( self.env_options.Q_HD[self.day][self.time_step] + g_tesc + g_tesd - g_FC - g_EL - self.env_options.P_solar_heat[self.day][self.time_step] + g_AC, 0 )
            )

        # reward
        reward = self.env_options.penalty1 * C1 + self.env_options.penalty2 * C2 + self.env_options.penalty3 * C3 + self.env_options.penalty4 * C4 + self.env_options.penalty5 * C5
        return reward, C1, C2 + C3 + C4 + C5            # 返回奖励，成本和违反约束水平


if __name__ == '__main__':
    env = HIES_Env()
    env_options = HIES_Options()
    state = env.reset()
    done = False
    i = 0
    states = []
    rewards = []
    p_t = [(env_options.b0 + env_options.b1 * env_options.T_tank_init) * (env_options.S_bss_init / env_options.V_tank) / (10**6)]
    states.append(env.states)
    while not done:
        actions = np.random.uniform(0, 1, size=7)
        scaled_action_indices = np.array([0, 1, 2, 4, 6], dtype = np.int32)
        for idx in scaled_action_indices:
            actions[idx] = actions[idx] * 2 - 1
        next_states, reward, done, violation, cost = env.step(actions)
        state = next_states
        states.append(env.states)
        p_t.append(env.get_decisions(actions)['p_t'])
        rewards.append(reward)
    
    S_hss = [state[6] for state in states]
    S_bss = [state[5] for state in states]
    hours = np.arange(len(S_hss))
    plt.step(hours, S_hss, where='mid', linestyle='-', color='k', label='m_H2')
    plt.step(hours, S_bss, where='mid', linestyle='-', color='r', label='S_bss') 
    # 添加图例和标签
    plt.xlabel('Hours')
    plt.ylabel('Storage Values')
    plt.legend()
    plt.show()

    p_low = [env_options.p_low for i in range(len(p_t))]
    p_high = [env_options.p_high for i in range(len(p_t))]

    print(p_low)

    plt.step(hours, p_t, where='mid', linestyle='-', color='k', label='p_H2')
    plt.plot(hours, p_low, 'b-')
    plt.plot(hours, p_high, 'r-')
    plt.xlabel('Hours')
    plt.ylabel('Pressure of Hydrogen Tank/bar')
    plt.show()




        
    













