
from DataProcess import ROPDataset
import pytorch_lightning as pl
from model.MLPROPModel import MLPROPModel
from model.ROPBitWearCoupledModel import ROPBitWearCoupledModel
from dataset.dataLoader import AllForOneROPDataset, ALLForOneROPDatasetBitSpecific, padding_drilling_bit_historical_data
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch 


# 这一句是为了设置默认的数据类型为float32，hmeng的电脑是mac 需要设置，正常不需要设置
torch.set_default_dtype(torch.float32)



def train_test_val_split():

    well_ids = ['SZ36-1-Q2','SZ36-1-Q3','SZ36-1-Q4','SZ36-1-Q6','SZ36-1-Q7','SZ36-1-Q8','SZ36-1-Q9','SZ36-1-Q10','SZ36-1-Q11','SZ36-1-Q13','SZ36-1-Q14',
                'SZ36-1-Q15','SZ36-1-Q16H','SZ36-1-Q17H','SZ36-1-Q18H','SZ36-1-Q19H','SZ36-1-Q20H','SZ36-1-Q21H','SZ36-1-Q22H','SZ36-1-Q24H'] # remove SZ36-1-Q23H 因为这个缺少一列 'DVER'

    train_well_ids = well_ids[:7]
    val_well_ids = well_ids[7:-1]
    test_well_ids = well_ids[-1:]


    train_well_file_path_list = [f'./data/ProcessedData/{well_id}.csv' for well_id in train_well_ids]
    val_well_file_path_list = [f'./data/ProcessedData/{well_id}.csv' for well_id in val_well_ids]
    test_well_file_path_list = [f'./data/ProcessedData/{well_id}.csv' for well_id in test_well_ids]

    return train_well_file_path_list, val_well_file_path_list, test_well_file_path_list






    return train_loader, test_loader, val_loader
def train_MLP_model():

    
    train_well_file_path_list, val_well_file_path_list, test_well_file_path_list = train_test_val_split()

    train_dataset = AllForOneROPDataset(train_well_file_path_list)
    val_dataset = AllForOneROPDataset(val_well_file_path_list, train_dataset.scaler)
    test_dataset = AllForOneROPDataset(test_well_file_path_list, train_dataset.scaler)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


    hparams = {
        'input_dim': len(train_dataset.input_col),
        'hidden_dim': 64,
        'learning_rate': 1e-3
    }

    model = MLPROPModel(hparams)

    # checkpoint_callback 用来保存模型，logger用来记录训练过程
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='MLPROPModel-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    logger = TensorBoardLogger('logs/', name='MLPROPModel')

    trainer = pl.Trainer(
        accelerator="mps",
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=20,
    )

    trainer.fit(model, train_loader, val_loader)
    test_result = trainer.test(model, test_loader)



def train_BitWearCoupled_model():

    train_well_file_path_list, val_well_file_path_list, test_well_file_path_list = train_test_val_split()

    train_dataset = ALLForOneROPDatasetBitSpecific(train_well_file_path_list)
    val_dataset = ALLForOneROPDatasetBitSpecific(val_well_file_path_list, train_dataset.scaler)
    test_dataset = ALLForOneROPDatasetBitSpecific(test_well_file_path_list, train_dataset.scaler)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=padding_drilling_bit_historical_data)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=padding_drilling_bit_historical_data)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, collate_fn=padding_drilling_bit_historical_data)

    hparams = {
        'input_dim': len(train_dataset.input_col),
        'hidden_dim': 64,
        'learning_rate': 1e-3,
        'dropout': 0.5
    }

    model = ROPBitWearCoupledModel(hparams)

    # checkpoint_callback 用来保存模型，logger用来记录训练过程
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        filename='ROPBitWearCoupledModel-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    logger = TensorBoardLogger('logs/', name='ROPBitWearCoupledModel')

    trainer = pl.Trainer(
        accelerator="mps",
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=20,
    )

    trainer.fit(model, train_loader, val_loader)
    test_result = trainer.test(model, test_loader)

if __name__ == "__main__":
    # train_MLP_model()
    train_BitWearCoupled_model()