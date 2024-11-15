# this line is added by juncai

import pandas as pd
import os

import numpy as np
from utils.visualisation import plot_wells_data
import matplotlib.pyplot as plt
import seaborn as sns

def process_drill_log(drill_log):
    """
    # 处理钻井日志数据
    """
    # Remove the first row (usually contains units or additional headers)
    drilling_log = drill_log.iloc[1:, :]

    # Set the header to the first row of the remaining data
    drilling_log.columns = drilling_log.iloc[0]

    # Remove the row that was used as header
    drilling_log = drilling_log.iloc[1:, :]
    return drilling_log.dropna(axis=1, how='all')


def load_and_preprocess_mudlogging_data(file_path):
    """
    # 加载并预处理录井数据
    """
    df = pd.read_csv(file_path, low_memory=False)
    units = df.iloc[0, :]
    df = df.iloc[1:, :]
    df["TIME"] = pd.to_datetime(df["TIME"])
    df = df.set_index("TIME").sort_index()
    df = df.astype(float)
    
    return df, units

def filter_the_pure_drilling_row(drill_log):
    """
    筛选出纯钻进的时间，这个时候根据 DMEA 井眼深度 来判断，如果DMEA相邻两行不等，下一行比上一行大，说明是正在钻进
    
    并且删除所有的数值为 0 的列，或者所有的数值为一样的列， 包括所有的值为空的情况
    """
    drill_log['DMEA'] = drill_log['DMEA'].astype(float)
    drill_log = drill_log[drill_log['DMEA'].diff() > 0]

    # 并且删除所有的数值为 0 的列，或者所有的为空的列
    drill_log = drill_log.dropna(axis=1, how='all')
    drill_log = drill_log.loc[:, (drill_log != 0).any(axis=0)]

    return drill_log


def fill_abnormal_values_with_closest_ones(df, column, lower_bound = -np.inf, higher_bound = np.inf):
    """
    Fill abnormal values in a column using the closest non-abnormal values.
    
    Parameters:
    df: pandas.DataFrame with the data
    column: str, the column to process
    lower_bound: float, the lower bound for the column values
    higher_bound: float, the higher bound for the column values
    
    Returns:
    pandas.DataFrame: Processed dataframe with filled values
    """
    # Create a copy of the data
    df_filled = df.copy()

    # Find positions where the column values are outside the bounds
    abnormal_mask = (df_filled[column] < lower_bound) | (df_filled[column] > higher_bound)

    # Set the abnormal values to NaN
    df_filled.loc[abnormal_mask, column] = np.nan
    
    # Get forward and backward filled values
    forward_filled = df_filled[column].ffill()
    backward_filled = df_filled[column].bfill()
    
    # Get indices of abnormal values
    abnormal_indices = df_filled.index[abnormal_mask]
    
    # Get indices where the column values are within the bounds
    valid_indices = df_filled.index[df_filled[column].notna()]
    
    for idx in abnormal_indices:
        # Find valid values before and after the current index
        forward_valid = valid_indices[valid_indices < idx]
        backward_valid = valid_indices[valid_indices > idx]
        
        # Get the nearest valid values
        if len(forward_valid) == 0:
            # If no forward values, use backward
            df_filled.at[idx, column] = backward_filled[idx]
        elif len(backward_valid) == 0:
            # If no backward values, use forward
            df_filled.at[idx, column] = forward_filled[idx]
        else:
            # Calculate distances to nearest valid values
            forward_distance = abs((idx - forward_valid[-1]))
            backward_distance = abs((backward_valid[0] - idx))
            
            # Choose the nearest value
            if forward_distance <= backward_distance:
                df_filled.at[idx, column] = forward_filled[idx]
            else:
                df_filled.at[idx, column] = backward_filled[idx]
    
    return df_filled


def add_drilling_bit_information(drill_log, well_name):
    """
    在录井数据中增加钻头信息

    参数：
    drill_log: pandas.DataFrame, 包含录井数据
    well_name: str, 井名

    返回：
    pandas.DataFrame: 包含钻头信息的录井数据

    """

    if well_name in ['SZ36-1-Q2','SZ36-1-Q3','SZ36-1-Q4','SZ36-1-Q6','SZ36-1-Q7','SZ36-1-Q8','SZ36-1-Q9','SZ36-1-Q10','SZ36-1-Q11','SZ36-1-Q13','SZ36-1-Q14']:
       BIT_path='./data/钻头总结/钻头总结2-14.xlsx'
    elif well_name in ['SZ36-1-Q15','SZ36-1-Q16H','SZ36-1-Q17H','SZ36-1-Q18H','SZ36-1-Q19H','SZ36-1-Q20H','SZ36-1-Q21H','SZ36-1-Q22H','SZ36-1-Q23H','SZ36-1-Q24H']:
       BIT_path='./data/钻头总结/钻头总结15-24.xlsx'
    else:
       BIT_path=[]
    drill_bit_summary = pd.read_excel(BIT_path)

    drill_bit_summary = drill_bit_summary[drill_bit_summary['井名'] == well_name]

    # add columns if PDC in the 钻头类型
    drill_bit_summary['PDC'] = drill_bit_summary['钻头类型'].apply(lambda x: 'PDC' in x)
    drill_bit_summary = drill_bit_summary[drill_bit_summary['PDC'] == True]

    for i in range(len(drill_bit_summary)):
        start_time = pd.Timestamp(drill_bit_summary.iloc[i]['入井时间']).tz_localize('UTC+08:00')
        end_time = pd.Timestamp(drill_bit_summary.iloc[i]['出井时间']).tz_localize('UTC+08:00')

        bit_diameter = drill_bit_summary.iloc[i]['尺寸(in)']

        # 将尺寸转换为毫米
        if '1/4' in bit_diameter:
            bit_diameter = int(bit_diameter.split(' ')[0]) * 25.4 + 25.4/4
        elif '1/2' in bit_diameter:
            bit_diameter = int(bit_diameter.split(' ')[0]) * 25.4 + 25.4/2        
        else:
            bit_diameter = int(bit_diameter) * 25.4
        
        drill_log.loc[start_time:end_time, 'BIT_DIAMETER'] = bit_diameter
        drill_log.loc[start_time:end_time, 'BIT_TFA'] = drill_bit_summary.iloc[i]['钻头TFA(in2)']

    # 仅仅保留有钻头信息的行
    drill_log = drill_log[drill_log['BIT_DIAMETER'].notna()]

    return drill_log

#井斜角计算函数
def calculate_incline(df):
    '''
    计算井斜角，单位为度，

    参数：
    df: pandas.DataFrame
    
    返回：
    pandas.DataFrame: 计算处理后的，处理后的数据, 包含井斜角
    '''
    # 导入相关库
    from scipy.ndimage import median_filter

    # 创建数据副本
    df_filled = df.copy()
    #创建辅助列，井深的差值与垂深的差值
    df_filled['diff1'] = df_filled['DMEA'].diff(periods=24)  
    df_filled['diff2'] = df_filled['DVER'].diff(periods=24)

    # 进行反正弦计算出角度
    df_filled['ANGLE'] = np.arccos(df_filled['diff2'] / df_filled['diff1'])
    # 填充缺失值
    df_filled['ANGLE'] = df_filled['ANGLE'].bfill()
    df_filled['ANGLE'] = df_filled['ANGLE'].ffill()
    # 进行中值滤波
    df_filled['ANGLE'] = median_filter(df_filled['ANGLE'], size=60)  
    # 转换为角度
    df_filled['ANGLE'] = df_filled['ANGLE'] * 180 / np.pi  
    # 移除辅助列
    df_filled.drop(columns=[ 'diff1', 'diff2'], inplace=True)

    return df_filled


#水力参数计算函数
def calculate_JET(df):
    '''
    计算射流速度、射流冲击力、射流水功率

    参数：
    df: pandas.DataFrame, 包含RPMA列的数据框，index为时间戳
    
    返回：
    pandas.DataFrame: 处理后的数据框
    '''
    print('length of df:',len(df))
    df_filled = df.copy()
    df_filled['BIT_DIAMETER']  = df_filled['BIT_DIAMETER'].astype(float)
    df_filled['BIT_TFA']  = df_filled['BIT_TFA'].astype(float)
    df_filled['MFIA']  = df_filled['MFIA'].astype(float)
    df_filled['MDIA']  = df_filled['MDIA'].astype(float)
    ##射流速度,单位m/s
    df_filled['BIT_JET_SPEED']=10*df_filled['MFIA']/df_filled['BIT_TFA']/6.4516/60

    ##射流冲击力,单位kN
    df_filled['BIT_JET_IMPACT_FORCE']=df_filled['MDIA']*(df_filled['MFIA']/60)**2/100/df_filled['BIT_TFA']/6.4516

    ##射流水功率,单位kW
    df_filled['BIT_JET_POWER']=df_filled['MDIA']*(df_filled['MFIA']/60)**3/20/(df_filled['BIT_TFA']*6.4516)**2

    print('length of df_filled:',len(df_filled))
    return df_filled
 

def smooth_the_ROPA(df, allowed_time_gap=300):
    """
    平滑ROPA数据，基于时间间隔对数据进行分段处理
    
    参数:
    df: DataFrame, 包含ROPA数据的数据框
    allowed_time_gap: int, 允许的最大时间间隔（秒）
    
    返回:
    DataFrame: 包含平滑后ROPA数据的数据框
    """
    # 复制数据框以避免修改原始数据
    df = df.copy()
    
    # 转换时间索引
    df['Time'] = pd.to_datetime(df.index)
    delta_time = df['Time'].diff()
    delta_time = delta_time.dt.total_seconds()
    df['breakpoints'] = delta_time > allowed_time_gap
    
    # 初始化平滑后的ROPA列
    df['ROPASMOOTH'] = np.nan
    
    # 使用索引值进行分段处理
    ropa_list = []
    start_idx = df.index[0]
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if not row['breakpoints']:
            ropa_list.append(row['ROPA'])
        else:
            # 如果遇到断点，处理之前收集的数据
            if ropa_list:  # 确保列表不为空
                end_idx = idx
                df.loc[start_idx:end_idx, 'ROPASMOOTH'] = np.mean(ropa_list)
                start_idx = idx
                ropa_list = [row['ROPA']]  # 开始新的分段，包含当前值
    
    # 处理最后一段数据
    if ropa_list:
        df.loc[start_idx:, 'ROPASMOOTH'] = np.mean(ropa_list)
    # 删除临时列
    df = df.drop(['Time', 'breakpoints'], axis=1)
    
    return df


def Calculate_ROPA(df, allowed_time_gap=300):
    """
    平滑ROPA数据，基于时间间隔对数据进行分段处理
    
    参数:
    df: DataFrame, 包含ROPA数据的数据框
    allowed_time_gap: int, 允许的最大时间间隔（秒）
    
    返回:
    DataFrame: 人工计算的ROPACAL数据的数据框
    """
    # 复制数据框以避免修改原始数据
    df = df.copy()
    
    # 转换时间索引
    df['Time'] = pd.to_datetime(df.index)
    delta_time = df['Time'].diff()
    delta_time = delta_time.dt.total_seconds()
    df['breakpoints'] = delta_time > allowed_time_gap
    
    # 初始化平滑后的ROPA列
    df['ROPACAL'] = np.nan
    
    # 使用索引值进行分段处理
    ropa_list = []
    start_idx = df.index[0]

    for i, (idx, row) in enumerate(df.iterrows()):
        if row['breakpoints']:
            start_depth = df.loc[start_idx, 'DMEA']
            end_idx = df.index[i-1]
            end_depth = df.loc[end_idx, 'DMEA']

            if start_depth == end_depth:
                calculatedROPA = 0
            calculatedROPA = (end_depth - start_depth) / (end_idx - start_idx).total_seconds() * 3600

            # 计算ROP
            df.loc[start_idx:end_idx, 'ROPACAL'] = calculatedROPA
            start_idx = idx
    
    start_depth = df.loc[start_idx, 'DMEA']
    end_depth = df.loc[df.index[-1], 'DMEA']
    # 计算最后一段ROP
    df.loc[start_idx:, 'ROPACAL'] = (end_depth - start_depth) / (df.index[-1] - start_idx).total_seconds() * 3600

    # 删除临时列
    df = df.drop(['Time', 'breakpoints'], axis=1)

    df['ROPACAL'] = df['ROPACAL'].bfill()
    df['ROPACAL'] = df['ROPACAL'].ffill()

    return df


def smooth_the_certain_column(df, column, allowed_time_gap = 300):

    # 复制数据框以避免修改原始数据
    df = df.copy()
    
    # 转换时间索引
    df['Time'] = pd.to_datetime(df.index)
    delta_time = df['Time'].diff()
    delta_time = delta_time.dt.total_seconds()
    df['breakpoints'] = delta_time > allowed_time_gap
    
    # 初始化平滑后的ROPA列
    df[f'{column}SMOOTH'] = np.nan
    
    # 使用索引值进行分段处理
    values_list = []
    start_idx = df.index[0]
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if not row['breakpoints']:
            values_list.append(row[column])
        else:
            # 如果遇到断点，处理之前收集的数据
            if values_list:  # 确保列表不为空
                end_idx = idx
                df.loc[start_idx:end_idx, f'{column}SMOOTH'] = np.median(values_list)
                start_idx = idx
                values_list = [row[column]]  # 开始新的分段，包含当前值
    
    # 处理最后一段数据
    if values_list:
        df.loc[start_idx:, f'{column}SMOOTH'] = np.median(values_list)
    # 删除临时列
    df[column] = df[f'{column}SMOOTH']
    df = df.drop(['Time', 'breakpoints', f'{column}SMOOTH'], axis=1)

    return df


def read_and_process_drill_muding_data(well_name):
    """
    读取录井数据和钻井数据
    """

    # 录井数据的目录
    mudlogging_files = os.listdir('./data/录井数据')
    mudlogging_files  = [file for file in mudlogging_files if file.endswith('.csv')]

    for file in mudlogging_files:
        #if file end with well_name.csv
        if file.endswith(well_name+'.csv'):
            file_path = './data/录井数据/'+file
            break

    # 读取录井数据
    mudlogging_data, _ = load_and_preprocess_mudlogging_data(file_path)

    # 筛选出纯钻进的行
    pure_drilling_mudlodding_data = filter_the_pure_drilling_row(mudlogging_data)
    
    # 选择需要的列
    selected_columns = ['DMEA', 'DVER', 'BITRUN', 'WOBA', 'RPMA', 'MFIA', 'MDIA', 'SPPA', 'ROPA', 'TQA']
    pure_drilling_mudlodding_data = pure_drilling_mudlodding_data[selected_columns]

    # 填充RPMA中的不合理的值
    pure_drilling_mudlodding_data = fill_abnormal_values_with_closest_ones(pure_drilling_mudlodding_data, column = 'RPMA', lower_bound=10)

    # 填充WOBA中不合理的值
    pure_drilling_mudlodding_data = fill_abnormal_values_with_closest_ones(pure_drilling_mudlodding_data, column = 'WOBA', lower_bound=0, higher_bound=20)

    # 在录井数据中增加钻头信息
    pure_drilling_mudlogging_data_with_bit_info = add_drilling_bit_information(pure_drilling_mudlodding_data, well_name)

    # 计算井斜角
    pure_drilling_mudlogging_data_with_bit_info = calculate_incline(pure_drilling_mudlogging_data_with_bit_info)

    # 计算水力参数
    pure_drilling_mudlogging_data_with_bit_info = calculate_JET(pure_drilling_mudlogging_data_with_bit_info)

    # 平滑ROPA数据
    pure_drilling_mudlogging_data_with_bit_info = smooth_the_ROPA(pure_drilling_mudlogging_data_with_bit_info, 300)

    # 计算ROP
    pure_drilling_mudlogging_data_with_bit_info = Calculate_ROPA(pure_drilling_mudlogging_data_with_bit_info, 300)

    # # smooth WOBA RPMA 
    # pure_drilling_mudlogging_data_with_bit_info = smooth_the_certain_column(pure_drilling_mudlogging_data_with_bit_info, 'WOBA', 300)
    # pure_drilling_mudlogging_data_with_bit_info = smooth_the_certain_column(pure_drilling_mudlogging_data_with_bit_info, 'RPMA', 300)

    return pure_drilling_mudlogging_data_with_bit_info



if __name__ == '__main__':

    # 读取 11 口井的钻井日志数据
    well_ids = ['SZ36-1-Q2','SZ36-1-Q3','SZ36-1-Q4','SZ36-1-Q6','SZ36-1-Q7','SZ36-1-Q8','SZ36-1-Q9','SZ36-1-Q10','SZ36-1-Q11','SZ36-1-Q13','SZ36-1-Q14',
                'SZ36-1-Q15','SZ36-1-Q16H','SZ36-1-Q17H','SZ36-1-Q18H','SZ36-1-Q19H','SZ36-1-Q20H','SZ36-1-Q21H','SZ36-1-Q22H','SZ36-1-Q23H','SZ36-1-Q24H']
    # well_ids = ['SZ36-1-Q13']
    
    for well_ind in well_ids:
        
        if well_ind == 'SZ36-1-Q23H':
            # "['DVER'] not in index"
            continue

        df_data = read_and_process_drill_muding_data(well_ind)
        # save preprocessed data into csv file into ./data/ProcessedData
        if os.path.exists('./data/ProcessedData') == False:
            os.makedirs('./data/ProcessedData')
        df_data.to_csv(f'./data/ProcessedData/{well_ind}.csv', index=True)