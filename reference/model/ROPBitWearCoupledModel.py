import pytorch_lightning as pl
import torch.nn as nn
import torch

# 这一句是为了设置默认的数据类型为float32，hmeng的电脑是mac 需要设置，正常不需要设置
torch.set_default_dtype(torch.float32)

# 这个类是一个简单的MLP模型，用来预测ROP
class ROPBitWearCoupledModel(pl.LightningModule):
    def __init__(self, params):
        super(ROPBitWearCoupledModel, self).__init__()

        self.params = params
        
        self.bitwearcoefmodel= BitWearCoef(params)

        self.newbitROP = NewBitROPModel(params)

    def forward(self, batch):
        
        drilling_bit_historical_data = batch['drilling_bit_historical_data']
        that_row_data = batch['that_row_data']
        historical_length = batch['historical_length']

        # 预测钻头的磨损系数
        bitwearcoef = self.bitwearcoefmodel(drilling_bit_historical_data, historical_length)

        # 预测新钻头的ROP
        newbitROP = self.newbitROP(that_row_data)

        # 钻头的ROP = 钻头的磨损系数 * 新钻头的ROP
        output = bitwearcoef * newbitROP
        
        return output.squeeze()
        
    def training_step(self, batch, batch_idx):
        y = batch['target']
        y_hat = self.forward(batch)

        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch['target']
        y_hat = self.forward(batch)

        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        y = batch['target']
        y_hat = self.forward(batch)
        
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params['learning_rate'])
    


class NewBitROPModel(pl.LightningModule):
    """
    这个模型用来预测新钻头的ROP
    """
    def __init__(self, params):
        super(NewBitROPModel, self).__init__()


        # 这个模型用来预测新钻头的ROP, 当前模型非常简单，需要进行优化，考虑可解释的模型
        self.params = params
        self.newbitmodel = nn.Sequential(
            nn.Linear(self.params['input_dim'], self.params['hidden_dim']),
            nn.ReLU(),
            nn.BatchNorm1d(self.params['hidden_dim']),
            nn.Linear(self.params['hidden_dim'], self.params['hidden_dim']),
            nn.ReLU(),
            nn.BatchNorm1d(self.params['hidden_dim']),
            nn.Linear(self.params['hidden_dim'], 1),
            # 最后这个用来保证ROP始终正的
            nn.ReLU()
        )
    
    def forward(self, that_row_data):

        return self.newbitmodel(that_row_data)


class BitWearCoef(pl.LightningModule):
    """
    这个模型用来预测钻头的磨损系数
    """
    def __init__(self, params):
        super(BitWearCoef, self).__init__()

        self.params = params
    
        self.bitwearcoefmodel = nn.Sequential(
            nn.Linear(self.params['input_dim'], self.params['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.params['dropout']),
            nn.Linear(self.params['hidden_dim'], self.params['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(self.params['dropout']),
            nn.Linear(self.params['hidden_dim'], 1),
            # 最后这个用来保证钻头的磨损始终正的
            nn.Softplus()
        )
    
    def forward(self, drilling_bit_historical_data, historical_length):
        
        historical_wear_coefficient = self.bitwearcoefmodel(drilling_bit_historical_data)
        wear_coefficient = self.gather_wear_coef(historical_wear_coefficient, historical_length)

        # 最后这个用来保证钻头的累积磨损因子始终位于0-1之间
        return nn.Sigmoid()(wear_coefficient)
    
    def gather_wear_coef(self, matrix, lengths):
        """
        仅仅收集前lengths个元素，因为后面是padding的，所以不需要考虑

        matrix: [batch_size, max_seq_len, feature_dim]
        lengths: [batch_size]
        """
        mask = torch.arange(matrix.shape[1], device=matrix.device)[None, :] < lengths[:, None]
        mask = mask.unsqueeze(-1)  # [32, 32, 1]
        # 应用掩码并求和
        masked_matrix = matrix * mask
        return masked_matrix.sum(dim=1)  # [32, 1]
    