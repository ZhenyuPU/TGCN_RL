import numpy as np
import pandas as pd
import os

def build_data_process(source_path, save_path):
    solar_radiation = pd.DataFrame()
    electricity_demand = pd.DataFrame()
    heating_demand = pd.DataFrame()
    cooling_demand = pd.DataFrame()
    T_a = pd.DataFrame()
    for filename in os.listdir(source_path):
        if filename.startswith('Building') and filename.endswith('.csv'):
            filepath = os.path.join(source_path, filename)
            df = pd.read_csv(filepath)
            electricity_demand[filename] = df['Equipment Electric Power [kWh]']
            heating_demand[filename] = df['DHW Heating [kWh]']
            cooling_demand[filename] = df['Cooling Load [kWh]']
            T_a[filename] = df['Indoor Temperature [C]']
        elif filename == 'weather.csv':
            filepath = os.path.join(source_path, filename)
            df = pd.read_csv(filepath)
            solar_radiation = df['Direct Solar Radiation [W/m2]']
    electricity_demand = pd.DataFrame(electricity_demand.sum(axis=1))[:365*24]
    heating_demand = pd.DataFrame(heating_demand.sum(axis=1))[:365*24]
    cooling_demand = pd.DataFrame(cooling_demand.sum(axis=1))[:365*24]
    T_a = pd.DataFrame(T_a.mean(axis=1))[:365*24] 
    solar_radiation = solar_radiation[:365*24] 
    solar_radiation = pd.DataFrame(solar_radiation.to_numpy().reshape(int(len(solar_radiation)/24), 24))

    electricity_demand = pd.DataFrame(electricity_demand.to_numpy().reshape(int(len(electricity_demand)/24), 24))

    heating_demand = pd.DataFrame(heating_demand.to_numpy().reshape(int(len(heating_demand)/24), 24))

    cooling_demand = pd.DataFrame(cooling_demand.to_numpy().reshape(int(len(cooling_demand)/24), 24))

    T_a = pd.DataFrame(T_a.to_numpy().reshape(int(len(T_a)/24), 24))

    solar_radiation.to_csv(save_path+'/solar_radiation.csv')
    electricity_demand.to_csv(save_path+'/electricity_demand.csv')
    heating_demand.to_csv(save_path+'/heating_demand.csv')
    cooling_demand.to_csv(save_path+'/cooling_demand.csv')
    T_a.to_csv(save_path+'/T_a.csv')





def price_process(source_path, save_path):
    df = pd.read_csv(source_path)
    electricity_price = df['system_energy_price_da'] / 100
    electricity_price = pd.DataFrame(electricity_price[:-1].to_numpy().reshape(int(len(electricity_price)/24), 24))
    electricity_price.to_csv(save_path+'/electricity_price.csv')



"""
    price solar e_demand h_demand c_demand T_a

    8760(365 * 24), 6(node num)

"""


def graph_data_contruct(source_path, save_path):
    price       =   pd.read_csv(source_path+'/electricity_price.csv', index_col=0).to_numpy().reshape(1, -1)
    solar_radiation =   pd.read_csv(source_path+'/solar_radiation.csv', index_col=0).to_numpy().reshape(1, -1)
    e_demand    =   pd.read_csv(source_path+'/electricity_demand.csv', index_col=0).to_numpy().reshape(1, -1)
    h_demand    =   pd.read_csv(source_path+'/heating_demand.csv', index_col=0).to_numpy().reshape(1, -1)
    c_demand    =   pd.read_csv(source_path+'/cooling_demand.csv', index_col=0).to_numpy().reshape(1, -1)
    T_a       =   pd.read_csv(source_path+'/T_a.csv', index_col=0).to_numpy().reshape(1, -1)

    dataset     =   np.concatenate((price, solar_radiation, e_demand, h_demand, c_demand, T_a), axis = 0).T

    np.savetxt(save_path + '/HIES_data.csv', dataset, delimiter=',')





if __name__ == '__main__':
    # citylearn_challenge_2021 
    build_data_process(source_path='datasets/citylearn_challenge_2021', save_path='datasets/data')
    price_process(source_path='datasets/electricity_price/da_hrl_lmps.csv', save_path='datasets/data')
    graph_data_contruct(source_path='datasets/data', save_path='datasets')



        


