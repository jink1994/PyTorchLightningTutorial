import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split,DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer

import json
import pandas as pd
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer


class BinaryClassificationModel(nn.Module):
    '''
    Model Structure: 
        SentenceBERT ---> Dropout ---> Dense Network
    
    '''
    def __init__(self,config): 
        super().__init__()
        
        #----------------------------
        #Bunch of stuff here depending on configuration file
        self.num_classes = config['num_classes'] 
        self.dropout_rate=config['dropout_rate']
        self.bert_model_label = config['bert_model_label']
        #----------------------------
        
        #Components that will be used in the dataflow function below
        self.huggingface_bert = SentenceTransformer(self.bert_model_label)
        self.bert_output_hiddensize = self.huggingface_bert.encode(["A sentence that doesn't matter"]).shape[1]
        #This sentence is just to get the model output size so that we can use it later
        self.self_defined_model= nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.bert_output_hiddensize,self.num_classes)
        )
    def forward(self,input_matrix):
        '''
        input_matrix shape: (batch_size,max_seq_len)
        output shape:       (batch_size,num_classes) #2 classes here since it's binary classification
        '''
        batch_size = input_matrix.shape[0]
        output = self.huggingface_bert.encode(input_matrix)
        output = self.self_defined_model(output)
        logits = output.view(batch_size,self.num_classes)
        #'logits' are by definition the raw output vectors for classification tasks
        return logits

        
#==================================================================
#The above is purely the model

#detach them for better inference

#The following is the classifier
#==================================================================

class Classifier(pl.LightningModule):
    def __init__(self,model,config):
        super().__init__()
        self.model=model
        self.batch_size = config['batch_size'] # this line of code is required to use auto batch size selection in Trainer()
    def training_step(self,batch,batch_idx):
        x,y=batch
        logits = self.model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        return pl.TrainResult(loss)
    def validation_step(self,batch,batch_idx):
        x,y = batch
        logits =self.model(x)
        result = pl.EvalResult()
        result.log('val_loss',loss)
        return result
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        return optimizer

class SarcasmDataModule(pl.LightningDataModule):
    def __init__(self,config):
        #----------------------------
        #data_path/batch_size/transform
        self.data_path1 = config['complete_path']
        self.data_path2 = config['incomplete_path']
        self.batch_size = config['batch_size']

        #----------------------------
    def prepare_data(self):
        '''Do the data download/tokenize here'''
        pass
    def setup(self,stage:str):
        '''
        data operations:
        -data splits train/val/test
        -create datasets
        -apply transform
        -etc
        '''
        #self.transform = transforms.Compose([
        #    transforms.ToTensor()
        #])
        #usually we have to do some transformation to the dataset here
        #but luckily the entry of our model is the pretrained model from hugginface
        #thus we can just use plain text
        df1 = pd.read_csv(self.data_path1)
        df2 = pd.read_csv(self.data_path2)
        df1['label']=pd.Series([1 for i in range(len(df1))])
        df2['label']=pd.Series([0 for i in range(len(df2))])
        self.dataset = pd.concat([df1[['sentences','label']],df2[['sentences','label']]]).values.tolist()[:240000]
        self.train_val,self.test = random_split(self.dataset,[200000,40000])
        self.train,self.val = random_split(self.train_val,[180000,20000])
        
    def train_dataloader(self):
        train = DataLoader(self.train, batch_size=self.batch_size)
        return train
    def val_dataloader(self): 
        val = DataLoader(self.val, batch_size=self.batch_size)
        return val
    def test_dataloader(self):
        test = DataLoader(self.test, batch_size=self.batch_size)
        return test


def cli_main(config_path='./config.json'):
    #------------------
    #args
    #------------------
    with open(config_path) as f:
        config = json.load(f)
    
    #------------------
    #model and data
    #------------------
    backbone = BinaryClassificationModel(config)
    model = Classifier(backbone,config)
    dm = SarcasmDataModule(config)
    dm.setup("fit")
    #------------------
    #training
    #------------------   
    trainer = Trainer(gpus=1,auto_scale_batch_size= False)
    trainer.fit(model,dm)
    #------------------
    #testing
    #------------------
    result = trainer.test(test_dataloaders=dm)
    print(result)

if __name__ == "__main__":
    cli_main()