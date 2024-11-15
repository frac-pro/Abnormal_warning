from torch.utils.data import Dataset
import pandas as pd

from sklearn.preprocessing import StandardScaler
import torch 
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class AllForOneROPDataset(Dataset):
    """
    这个数据集用来加载录井数据，是按照整口井来进行加载的，并且用每一行的特征来预测该行所对应的ROP
    
    """
    def __init__(self, file_path_list, scaler = None):
        self.df = [pd.read_csv(file_path, index_col=0) for file_path in file_path_list]

        self.all_data = pd.concat(self.df)
        
        # 模型的输入特征是DMEA,DVER,BITRUN,WOBA,RPMA,MFIA,MDIA,SPPA,BIT_DIAMETER,BIT_TFA,ANGLE,BIT_JET_SPEED,BIT_JET_IMPACT_FORCE,BIT_JET_POWER
        self.input_col = ['DMEA','DVER','BITRUN','WOBA','RPMA','MFIA','MDIA','SPPA','BIT_DIAMETER','BIT_TFA','ANGLE','BIT_JET_SPEED','BIT_JET_IMPACT_FORCE','BIT_JET_POWER']
        self.input_feat = self.all_data[self.input_col]
        
        self.target_col = ['ROPASMOOTH']#,'ROPACAL','ROPA'
        
        self.scaler = scaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.input_feat)
        
        self.scalered_input_feat = self.scaler.transform(self.input_feat)
        
        self.output_feat = self.all_data[self.target_col]

    def __len__(self):
        return self.all_data.shape[0]
    
    def __getitem__(self, idx):
        
        # set torch.float32
        return torch.tensor(self.scalered_input_feat[idx], dtype=torch.float32), torch.tensor(self.output_feat.iloc[idx].values, dtype=torch.float32)

        

class ALLForOneROPDatasetBitSpecific(Dataset):
    """
    这个数据集也是用来加载数据集的，用前期钻井的信息来预测新的井，与AllForOneROPDataset的区别是，这个数据集加载的数据，不仅仅包含特定行的数据，还包含这个钻头钻到这个行的时候的历史信息
    
    """
    def __init__(self, file_path_list, scaler = None):
        # 原始的数据
        self.df = [pd.read_csv(file_path, index_col=0) for file_path in file_path_list]
        # set the index to timedataformate
        for i in range(len(self.df)):
            self.df[i].index = pd.to_datetime(self.df[i].index)

        self._collectd_drilling_data_with_bit(10)
        
        # 将所有的数据合并到一起,这个是为了方便归一化
        self.collected_data = pd.concat(self.bit_specific_mudlog_data)
        
        # 模型的输入特征是DMEA,DVER,BITRUN,WOBA,RPMA,MFIA,MDIA,SPPA,BIT_DIAMETER,BIT_TFA,ANGLE,BIT_JET_SPEED,BIT_JET_IMPACT_FORCE,BIT_JET_POWER
        self.input_col = ['DMEA','DVER','BITRUN','WOBA','RPMA', 'TQA','MFIA','MDIA','SPPA','BIT_DIAMETER','BIT_TFA','ANGLE','BIT_JET_SPEED','BIT_JET_IMPACT_FORCE','BIT_JET_POWER']
        self.input_feat = self.collected_data[self.input_col]
        self.scaler = scaler
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.input_feat)
        
        self.target_col = ['ROPASMOOTH']#,'ROPACAL','ROPA'
        self.bit_specific_ROPA = [df[self.target_col].values for df in self.bit_specific_mudlog_data]

        # 将所有的数据进行归一化
        self.bit_specific_mudlog_data = [df[self.input_col] for df in self.bit_specific_mudlog_data]
        self.scalered_bit_specific_mudlog_data = [self.scaler.transform(df) for df in self.bit_specific_mudlog_data]


        self.target_col = ['ROPASMOOTH']#,'ROPACAL','ROPA'

    def __len__(self):

        return np.sum(self.bit_specific_mudlog_data_len)
    

    def __getitem__(self, idx):
        
        # 根据全局索引找到对应的DataFrame和局部行号
        df_index, local_row_index = self.locate_index(idx)

        # 获得历史数据和当前数据
        drilling_bit_historical_data = self.scalered_bit_specific_mudlog_data[df_index][:local_row_index+1]
        that_row_data = self.scalered_bit_specific_mudlog_data[df_index][local_row_index]

        target = self.bit_specific_ROPA[df_index][local_row_index]
        
        return {
            'drilling_bit_historical_data': torch.tensor(drilling_bit_historical_data, dtype=torch.float32),
            'that_row_data': torch.tensor(that_row_data, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'historical_length': len(drilling_bit_historical_data)  # 记录钻头历史的实际长度
        }
    
    def locate_index(self, global_index):
        """
        使用numpy快速定位全局索引对应的DataFrame和局部行号
        
        Args:
            global_index: 全局索引值
            
        Returns:
            tuple: (df_index, local_row_index) DataFrame的索引和对应的行号
        """
        import numpy as np
        cumsum = np.cumsum([0] + self.bit_specific_mudlog_data_len)
        df_index = np.searchsorted(cumsum, global_index, side='right') - 1
        
        if df_index < 0 or df_index >= len(self.bit_specific_mudlog_data_len):
            return None
            
        local_row_index = global_index - cumsum[df_index]
        return df_index, local_row_index
    
    def _collectd_drilling_data_with_bit(self, min_rows):
        # 按照钻头进行数据的切分，每一个dataframe代表一个钻头的数据，这个钻头工刚开始使用新的，一直到出井的数据
        self.bit_specific_mudlog_data = []
        for i in range(len(self.df)):
            df = self.df[i]
            diff_bitrun = df['BITRUN'].diff()
            is_new_bit = diff_bitrun < 0
            new_bit_index = is_new_bit[is_new_bit].index

            if len(new_bit_index) == 0:
                # 这个代表一个钻头从头钻到尾
                self.bit_specific_mudlog_data.append(df)
            else:
                # new_bit_index是时间，在首个时间之前的所有的数据是一个新钻头的开始
                self.bit_specific_mudlog_data.append(df.loc[:new_bit_index[0]].iloc[:-1])

                for j in range(1, len(new_bit_index)):
                    self.bit_specific_mudlog_data.append(df.loc[new_bit_index[j-1]:new_bit_index[j]].iloc[:-1])
                
                self.bit_specific_mudlog_data.append(df.loc[new_bit_index[-1]:])

        # 提出行数小于10行的数据，因为原始数据的记录是存在一定的问题，BITRUN代表所用的钻头钻的深度
        self.bit_specific_mudlog_data = [df for df in self.bit_specific_mudlog_data if df.shape[0] > min_rows]

        # 记录每一个钻头的数据的长度
        self.bit_specific_mudlog_data_len = [df.shape[0] for df in self.bit_specific_mudlog_data]


def padding_drilling_bit_historical_data(batch):
    """
    对于ALLForOneROPDatasetBitSpecific中的drilling_bit_historical_data来说，每一个batch中的历史数据的长度是不一样的，所以需要对其进行padding
    这个方法在DataLoader中使用
    
    """
    # 提取batch中的各个组件
    historical_data = [item['drilling_bit_historical_data'] for item in batch]
    that_row_data = torch.stack([item['that_row_data'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    seq_lengths = torch.tensor([item['historical_length'] for item in batch])
    
    # 对历史数据进行padding
    padded_historical_data = pad_sequence(historical_data, batch_first=True)
    
    return {
        'drilling_bit_historical_data': padded_historical_data,  # [batch_size, max_seq_len, feature_dim]
        'that_row_data': that_row_data,  # [batch_size, feature_dim]
        'target': targets,  # [batch_size]
        'historical_length': seq_lengths  # [batch_size]
    }






if __name__ == '__main__':
    well_ids = ['SZ36-1-Q2','SZ36-1-Q3','SZ36-1-Q4','SZ36-1-Q6','SZ36-1-Q7','SZ36-1-Q8','SZ36-1-Q9','SZ36-1-Q10','SZ36-1-Q11','SZ36-1-Q13','SZ36-1-Q14',
                'SZ36-1-Q15','SZ36-1-Q16H','SZ36-1-Q17H','SZ36-1-Q18H','SZ36-1-Q19H','SZ36-1-Q20H','SZ36-1-Q21H','SZ36-1-Q22H','SZ36-1-Q24H']

    train_well_ids = well_ids[:7]
    test_well_ids = well_ids[7:]

    train_well_file_path_list = [f'./data/ProcessedData/{well_id}.csv' for well_id in train_well_ids]
    test_well_file_path_list = [f'./data/ProcessedData/{well_id}.csv' for well_id in test_well_ids]

    # train_dataset = AllForOneROPDataset(train_well_file_path_list)
    # test_dataset = AllForOneROPDataset(test_well_file_path_list, train_dataset.scaler)

    train_dataset = ALLForOneROPDatasetBitSpecific(train_well_file_path_list)
    test_dataset = ALLForOneROPDatasetBitSpecific(test_well_file_path_list, train_dataset.scaler)

    print(train_dataset[4])

