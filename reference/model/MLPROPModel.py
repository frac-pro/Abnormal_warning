import pytorch_lightning as pl
import torch.nn as nn
import torch

# 这一句是为了设置默认的数据类型为float32，hmeng的电脑是mac 需要设置，正常不需要设置
torch.set_default_dtype(torch.float32)

# 这个类是一个简单的MLP模型，用来预测ROP
class MLPROPModel(pl.LightningModule):
    def __init__(self, params):
        super(MLPROPModel, self).__init__()

        self.params = params
        
        self.linear1 = nn.Linear(self.params['input_dim'], self.params['hidden_dim'])
        self.relu = nn.ReLU()
        self.output = nn.Linear(self.params['hidden_dim'], 1)


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.output(x)
        return x.squeeze()
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params['learning_rate'])
    