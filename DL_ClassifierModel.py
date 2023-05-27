import numpy as np
import pandas as pd
import torch,time,os,pickle,random
from torch import nn as nn
from nnLayer import *
from metrics import *
from collections import Counter,Iterable
from sklearn.model_selection import StratifiedKFold,KFold
from torch.backends import cudnn
from tqdm import tqdm
from torchvision import models
from pytorch_lamb import lamb
from torch.utils.data import DataLoader,Dataset
from functools import reduce

def dict_to_device(data, device):
    for k in data:
        if isinstance(data[k], dict):
            data[k] = dict_to_device(data[k], device)
        elif data[k] is not None:
            data[k] = data[k].to(device)
    return data

class BaseModel:
    def __init__(self, model):
        pass
    def calculate_y_logit(self, X):
        pass
    def calculate_y_prob(self, X):
        pass
    def calculate_y(self, X):
        pass
    def calculate_y_prob_by_iterator(self, dataStream):
        pass
    def calculate_y_by_iterator(self, dataStream):
        pass
    def calculate_loss(self, X, Y):
        pass
    def train(self, optimizer, trainDataSet, validDataSet=None, 
              batchSize=256, maxSteps=1000000, evalSteps=100, earlyStop=10, saveSteps=-1, 
              isHigherBetter=False, metrics="LOSS", report=["LOSS"],   
              savePath='model', dataLoadNumWorkers=0, pinMemory=False, 
              warmupSteps=0, SEED=0, doEvalTrain=True, doEvalValid=True, prefetchFactor=16):
        metrictor = self.metrictor if hasattr(self, "metrictor") else Metrictor()
        device = next(self.model.parameters()).device
        # schedulerRLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if isHigherBetter else 'min', factor=0.5, patience=20, verbose=True)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        stop = False

        decaySteps = maxSteps - warmupSteps
        schedulerRLR = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i:i/warmupSteps if i<warmupSteps else (decaySteps-(i-warmupSteps))/decaySteps)

        trainSampler = None # if self.mode<=0 else torch.utils.data.distributed.DistributedSampler(trainDataSet, shuffle=True, seed=SEED)
        trainStream = DataLoader(trainDataSet, batch_size=batchSize, drop_last=True, shuffle=True if trainSampler is None else False, num_workers=dataLoadNumWorkers, pin_memory=pinMemory, collate_fn=self.collateFunc, sampler=trainSampler)
        evalTrainStream = DataLoader(trainDataSet, batch_size=batchSize, shuffle=False, num_workers=dataLoadNumWorkers, pin_memory=pinMemory, collate_fn=self.collateFunc, sampler=trainSampler, prefetch_factor=prefetchFactor)

        if validDataSet is not None: 
            validSampler = None # if self.mode<=0 else torch.utils.data.distributed.DistributedSampler(validDataSet, shuffle=False, seed=SEED)
            evalValidStream = DataLoader(validDataSet, batch_size=batchSize, shuffle=False, num_workers=dataLoadNumWorkers, pin_memory=pinMemory, collate_fn=self.collateFunc, sampler=validSampler, prefetch_factor=prefetchFactor)
        
        mtc,bestMtc,stopSteps = 0.0,0.0 if isHigherBetter else 9999999999,0
        e,locStep = 0,0

        while True:
            e += 1
            print(f"Training the step {locStep} at epoch {e}'s data...")
            self.to_train_mode()
            pbar = tqdm(trainStream)
            for data in pbar:
                data = dict_to_device(data, device=device)
                loss = self._train_step(data, optimizer)
                schedulerRLR.step()
                pbar.set_description(f"Training Loss: {loss}; Learning rate: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}; Stop rounds: {stopSteps}")
                locStep += 1
                if locStep>maxSteps:
                    print(f'Reach the max steps {maxSteps}... break...')
                    stop = True
                    break
                if (validDataSet is not None) and (locStep%evalSteps==0):
                    if True: # ((self.mode>0 and torch.distributed.get_rank() == 0) or self.mode==0):
                        print(f'========== Step:{locStep} at Epoch: {e} ==========')
                        with torch.no_grad():
                            self.to_eval_mode()
                            if doEvalTrain:
                                print(f'[Total Train]',end='')
                                data = self.calculate_y_prob_by_iterator(evalTrainStream, showProgress=False)
                                metrictor.set_data(data)
                                metrictor(report)
                            if doEvalValid:
                                print(f'[Total Valid]',end='')
                                data = self.calculate_y_prob_by_iterator(evalValidStream, showProgress=False)
                                metrictor.set_data(data)
                                res = metrictor(report)
                                mtc = res[metrics]
                                print('=================================')
                                if (mtc>bestMtc and isHigherBetter) or (mtc<bestMtc and not isHigherBetter):
                                    if True: # (self.mode>0 and torch.distributed.get_rank() == 0) or self.mode==0:                
                                       print(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                                       bestMtc = mtc
                                       self.save("%s.pkl"%savePath, e, bestMtc)
                                    stopSteps = 0
                                else:
                                    stopSteps += 1
                                    if stopSteps>=earlyStop:
                                        print(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e+1}, stop training.')
                                        stop = True
                                        break
                            self.to_train_mode()
                if saveSteps>0 and locStep%saveSteps==0:
                    self.save(f"%s_step{locStep}.pkl"%savePath, e+1, -1)
            if stop:
                break
        if True: # (self.mode>0 and torch.distributed.get_rank() == 0) or self.mode==0:
            with torch.no_grad():
                try:
                    self.load("%s.pkl"%savePath)
                    os.rename("%s.pkl"%savePath, "%s_%s.pkl"%(savePath, ("%.3lf"%bestMtc)[2:]))
                except:
                    print('No model saved...')
                self.to_eval_mode()
                print(f'============ Result ============')
                print(f'[Total Train]',end='')
                st = time.time()
                data = self.calculate_y_prob_by_iterator(evalTrainStream, showProgress=True)
                et = time.time()
                metrictor.set_data(data)
                metrictor(report)
                print(f'[Total Valid]',end='')
                data = self.calculate_y_prob_by_iterator(evalValidStream, showProgress=True)
                metrictor.set_data(data)
                res = metrictor(report)
                #metrictor.each_class_indictor_show(dataClass.id2lab)
                print(f'FPS: {len(trainDataSet) / (et-st):.3f}\n')
                print(f'================================')
                return res
    def to_train_mode(self):
        self.model.train()  #set the module in training mode
        if self.collateFunc is not None:
            if ('Linformer' not in self.model.__class__.__name__) and ('BigBird2' not in self.model.__class__.__name__):
                self.collateFunc.eval = False
            else:
                self.collateFunc.eval = True
    def to_eval_mode(self):
        self.model.eval()
        if self.collateFunc is not None:
            self.collateFunc.eval = True
    def _train_step(self, data, optimizer):
        optimizer.zero_grad()
        loss = self.calculate_loss(data)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
        optimizer.step()
        return loss
    def save(self, path, epochs, bestMtc=None):
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc, 'model':self.model.state_dict()}
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, key=None, map_location=None):
        parameters = torch.load(path, map_location=map_location)
        
        if key is None:
            self.model.load_state_dict(parameters['model'])
        else:
            tmp = parameters['model']
            for k in list(tmp.keys()):
                if key not in k:
                    tmp.pop(k)
                else:
                    tmp[k.replace('module.','').replace(f'{key}.','')] = tmp.pop(k)
            getattr(self.model,key).load_state_dict(tmp)

        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc']))

from torch.cuda.amp import autocast, GradScaler
class SequenceClassification(BaseModel):
    def __init__(self, model, collateFunc=None, AMP=False, criterion=None):
        self.model = model
        self.collateFunc = collateFunc
        self.metrictor = Metrictor()
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.AMP = AMP
        if AMP:
            self.scaler = GradScaler()
    def _train_step(self, data, optimizer):
        optimizer.zero_grad()
        if self.AMP:
            with autocast():
                loss = self.calculate_loss(data)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss = self.calculate_loss(data)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
        return loss
    def calculate_y_logit(self, data):
        if self.AMP:
            with autocast():
                return self.model(data)
        else:
            return self.model(data)
    def calculate_y_prob(self, data):
        Y_pre = self.calculate_y_logit(data)['y_logit']
        return {'y_prob':F.softmax(Y_pre, dim=-1)}
    def calculate_y(self, data):
        Y_pre = self.calculate_y_logit(data)['y_logit']
        return {'y_pre':F.argmax(Y_pre, dim=-1)}
    def calculate_loss_by_iterator(self, dataStream):
        loss,cnt = 0,0
        for data in dataStream:
            loss += self.calculate_loss(data) * len(data['tokenizedLabArr'])
            cnt += len(data['tokenizedLabArr'])
        return loss / cnt
    def calculate_y_prob_by_iterator(self, dataStream, showProgress=False):
        device = next(self.model.parameters()).device
        YArr,Y_preArr = [],[]
        if showProgress:
            dataStream = tqdm(dataStream)
        for data in dataStream:
            data = dict_to_device(data, device=device)
            Y_pre,Y = self.calculate_y_prob(data)['y_prob'].detach().cpu().data.numpy(),data['tokenizedLabArr'].detach().cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr,Y_preArr = np.concatenate(YArr, axis=0).astype('int32'),np.concatenate(Y_preArr, axis=0).astype('float32')
        return {'y_prob':Y_preArr, 'y_true':YArr}
    def calculate_y_by_iterator(self, dataStream):
        tmp = self.calculate_y_prob_by_iterator(dataStream)
        Y_preArr, YArr = tmp['y_prob'], torch.argmax(tmp['y_true'], dim=-1)
        return {'y_pre':Y_preArr, 'y_true':YArr}
    def calculate_loss(self, data):
        out = self.calculate_y_logit(data)
        Y = data['tokenizedLabArr'] # .reshape(-1)
        Y_logit = out['y_logit'] # .reshape(-1)
        loss = self.criterion(Y_logit, Y)
        if 'add_loss' in out:
            loss += out['add_loss']
        return loss

class SequenceMultiLabelClassification(SequenceClassification):
    def __init__(self, model, collateFunc=None, AMP=False, criterion=None):
        self.model = model
        self.collateFunc = collateFunc
        self.metrictor = Metrictor()
        self.criterion = nn.MultiLabelSoftMarginLoss() if criterion is None else criterion
        self.AMP = AMP
        if AMP:
            self.scaler = GradScaler()
    def calculate_y_prob(self, data):
        Y_pre = self.calculate_y_logit(data)['y_logit']
        return {'y_prob':F.sigmoid(Y_pre)}

class BaselineCNN(nn.Module):
    def __init__(self, classNum, 
                 embedding, 
                 hiddenSize=64, contextSizeList=[1,5,25], 
                 embDropout=0.2, hdnDropout=0.15):
        super(BaselineCNN, self).__init__()
        embSize = embedding.shape[1]

        self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.cnn = TextCNN(embSize, hiddenSize, contextSizeList, reduction='pool', actFunc=nn.ReLU, ln=True, name='cnn')
        self.fcLinear = MLP(hiddenSize*len(contextSizeList), classNum, dropout=hdnDropout, inDp=True, name='fcLinear')

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = self.cnn(x) # => batchSize × hiddenSize*scaleNum

        return {'y_logit':self.fcLinear(x)}

class BaselineRNN(nn.Module):
    def __init__(self, classNum, 
                 embedding, 
                 hiddenSize=64, 
                 embDropout=0.2, hdnDropout=0.15):
        super(BaselineRNN, self).__init__()
        embSize = embedding.shape[1]

        self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.rnn = TextLSTM(embSize, hiddenSize, ln=True, reduction='pool', name='rnn')
        self.fcLinear = MLP(hiddenSize*2, classNum, dropout=hdnDropout, inDp=True, name='fcLinear')

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = self.rnn(x) # => batchSize × hiddenSize*2

        return {'y_logit':self.fcLinear(x)}

class BaselineTransformer(nn.Module):
    def __init__(self, classNum, seqMaxLen, LN,
                 embedding, 
                 L=6, H=512, A=8, fcSize=1024, 
                 embDropout=0.2, hdnDropout=0.15):
        super(BaselineTransformer, self).__init__()

        if isinstance(embedding, VisioEmbedding):
            embSize = embedding.embedding[0].out_channels
            self.embedding = embedding
        elif isinstance(embedding, nn.Linear):
            embSize = embedding.weight.shape[0]
            self.embedding = embedding
        elif isinstance(embedding, InputCNN):
            embSize = H
            self.embedding = embedding
        else:
            embSize = embedding.shape[1]
            self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.posEmb = nn.Parameter(torch.tensor([[np.sin(pos/10000**(i/embSize)) if i%2==0 else np.cos(pos/10000**((i-1)/embSize)) for i in range(embSize)] for pos in range(seqMaxLen)], dtype=torch.float32), requires_grad=False) # seqLen × feaSize

        if LN=='post':
            self.transformer = TransformerLayers_PostLN(layersNum=L, feaSize=embSize, dk=H//A, multiNum=A, 
                                                        dropout=hdnDropout)
        elif LN=='pre':
            self.transformer = TransformerLayers_PreLN(layersNum=L, feaSize=embSize, dk=H//A, multiNum=A, 
                                                       dropout=hdnDropout)
        elif LN=='realformer':
            self.transformer = TransformerLayers_Realformer(layersNum=L, feaSize=embSize, dk=H//A, multiNum=A,
                                                            hdnDropout=hdnDropout)

        self.fcLinear = nn.Sequential(
                            nn.Linear(embSize*2, fcSize),
                            nn.BatchNorm1d(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = x+self.posEmb[:x.shape[1]]/3
        x = self.transformer(x, data['maskPAD'].unsqueeze(1)&data['maskPAD'].unsqueeze(2))[0] # => batchSize × seqLen × embSize
        m = data['maskPAD'].int()

        x_max,_ = torch.max(x+(1-m).unsqueeze(dim=-1)*(-1e10), dim=1)
        x_mean = (x*m.unsqueeze(dim=-1)).sum(dim=1) / m.sum(dim=1).unsqueeze(dim=-1)
        # x,_ = torch.max(x, dim=1)

        return {'y_logit':self.fcLinear(torch.cat([x_max,x_mean], dim=-1))}

class Singularformer(nn.Module):
    def __init__(self, classNum, seqMaxLen, 
                 embedding, 
                 r=64, linearLevel='l1',
                 L=6, H=512, A=8, fcSize=1024, 
                 embDropout=0.2, hdnDropout=0.15, gama=0.01):
        super(Singularformer, self).__init__()
        
        if isinstance(embedding, VisioEmbedding):
            embSize = embedding.embedding[0].out_channels
            self.embedding = embedding
        elif isinstance(embedding, nn.Linear):
            embSize = embedding.weight.shape[0]
            self.embedding = embedding
        elif isinstance(embedding, InputCNN):
            embSize = H
            self.embedding = embedding
        else:
            embSize = embedding.shape[1]
            self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.posEmb = nn.Parameter(torch.tensor([[np.sin(pos/10000**(i/embSize)) if i%2==0 else np.cos(pos/10000**((i-1)/embSize)) for i in range(embSize)] for pos in range(seqMaxLen)], dtype=torch.float32), requires_grad=False) # seqLen × feaSize
        # self.posEmb = nn.Embedding(seqMaxLen, embSize)

        self.pseformer = SingularformerLayers(layersNum=L, feaSize=embSize, dk=H//A, multiNum=A, r=r, linearLevel=linearLevel,
                                              hdnDropout=hdnDropout)
        self.fcLinear = nn.Sequential(
                            nn.Linear(embSize*2, fcSize),
                            nn.BatchNorm1d(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )
        self.gama = gama

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = x + self.posEmb[:x.shape[1]]/3
        # x = x + self.posEmb.weight[:x.shape[1]]/3
        x,_,_,_,_,addLoss = self.pseformer(x, data['maskPAD'].unsqueeze(1)&data['maskPAD'].unsqueeze(2)) # => batchSize × seqLen × embSize
        m = data['maskPAD'].int()
        x_max,_ = torch.max(x+(1-m).unsqueeze(dim=-1)*(-1e10), dim=1)
        x_mean = (x*m.unsqueeze(dim=-1)).sum(dim=1) / m.sum(dim=1).unsqueeze(dim=-1)
        # x,_ = torch.max(x, dim=1)

        return {'y_logit':self.fcLinear(torch.cat([x_max,x_mean], dim=-1)), 'add_loss':addLoss*self.gama}

class Pseformer(nn.Module):
    def __init__(self, classNum, seqMaxLen, 
                 embedding, 
                 pseTknNum=64,
                 L=6, H=512, A=8, fcSize=1024, 
                 embDropout=0.2, hdnDropout=0.15, gama=0.01):
        super(Pseformer, self).__init__()
        
        if isinstance(embedding, VisioEmbedding):
            embSize = embedding.embedding[0].out_channels
            self.embedding = embedding
        elif isinstance(embedding, nn.Linear):
            embSize = embedding.weight.shape[0]
            self.embedding = embedding
        elif isinstance(embedding, InputCNN):
            embSize = H
            self.embedding = embedding
        else:
            embSize = embedding.shape[1]
            self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.posEmb = nn.Parameter(torch.tensor([[np.sin(pos/10000**(i/embSize)) if i%2==0 else np.cos(pos/10000**((i-1)/embSize)) for i in range(embSize)] for pos in range(seqMaxLen)], dtype=torch.float32), requires_grad=False) # seqLen × feaSize
        # self.posEmb = nn.Embedding(seqMaxLen, embSize)

        self.pseformer = TransformerLayers_Pseudoformer(layersNum=L, feaSize=embSize, dk=H//A, multiNum=A, pseudoTknNum=pseTknNum, 
                                                        hdnDropout=hdnDropout)
        self.fcLinear = nn.Sequential(
                            nn.Linear(embSize*2, fcSize),
                            nn.BatchNorm1d(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )
        self.gama = gama

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = x + self.posEmb[:x.shape[1]]/3
        # x = x + self.posEmb.weight[:x.shape[1]]/3
        x,_,_,_,_,addLoss = self.pseformer(x, data['maskPAD'].unsqueeze(1)&data['maskPAD'].unsqueeze(2)) # => batchSize × seqLen × embSize
        m = data['maskPAD'].int()
        x_max,_ = torch.max(x+(1-m).unsqueeze(dim=-1)*(-1e10), dim=1)
        x_mean = (x*m.unsqueeze(dim=-1)).sum(dim=1) / m.sum(dim=1).unsqueeze(dim=-1)
        # x,_ = torch.max(x, dim=1)

        return {'y_logit':self.fcLinear(torch.cat([x_max,x_mean], dim=-1)), 'add_loss':addLoss*self.gama}

class Pseformer2(nn.Module):
    def __init__(self, classNum, seqMaxLen, 
                 embedding, 
                 pseTknNum=64,
                 L=6, H=512, A=8, fcSize=1024, 
                 embDropout=0.2, hdnDropout=0.15, gama=0.01):
        super(Pseformer2, self).__init__()
        
        if isinstance(embedding, VisioEmbedding):
            embSize = embedding.embedding[0].out_channels
            self.embedding = embedding
        elif isinstance(embedding, nn.Linear):
            embSize = embedding.weight.shape[0]
            self.embedding = embedding
        elif isinstance(embedding, InputCNN):
            embSize = H
            self.embedding = embedding
        else:
            embSize = embedding.shape[1]
            self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.posEmb = nn.Parameter(torch.tensor([[np.sin(pos/10000**(i/embSize)) if i%2==0 else np.cos(pos/10000**((i-1)/embSize)) for i in range(embSize)] for pos in range(seqMaxLen)], dtype=torch.float32), requires_grad=False) # seqLen × feaSize
        # self.posEmb = nn.Embedding(seqMaxLen, embSize)

        self.pseformer = TransformerLayers_Pseudoformer2(layersNum=L, feaSize=embSize, dk=H//A, multiNum=A, pseudoTknNum=pseTknNum, 
                                                         hdnDropout=hdnDropout)
        self.fcLinear = nn.Sequential(
                            nn.Linear(embSize*2, fcSize),
                            nn.BatchNorm1d(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )
        self.gama = gama

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = x + self.posEmb[:x.shape[1]]/3
        # x = x + self.posEmb.weight[:x.shape[1]]/3
        x,_,_,_,_,addLoss = self.pseformer(x, data['maskPAD'].unsqueeze(1)&data['maskPAD'].unsqueeze(2)) # => batchSize × seqLen × embSize
        m = data['maskPAD'].int()
        x_max,_ = torch.max(x+(1-m).unsqueeze(dim=-1)*(-1e10), dim=1)
        x_mean = (x*m.unsqueeze(dim=-1)).sum(dim=1) / m.sum(dim=1).unsqueeze(dim=-1)
        # x,_ = torch.max(x, dim=1)

        return {'y_logit':self.fcLinear(torch.cat([x_max,x_mean], dim=-1)), 'add_loss':addLoss*self.gama}

class cusLinformer(nn.Module):
    def __init__(self, classNum, seqMaxLen, 
                 embedding, 
                 pseTknNum=64,
                 L=6, H=512, A=8, fcSize=1024,
                 embDropout=0.2, hdnDropout=0.15):
        super(cusLinformer, self).__init__()

        if isinstance(embedding, VisioEmbedding):
            embSize = embedding.embedding[0].out_channels
            self.embedding = embedding
        elif isinstance(embedding, nn.Linear):
            embSize = embedding.weight.shape[0]
            self.embedding = embedding
        elif isinstance(embedding, InputCNN):
            embSize = H
            self.embedding = embedding
        else:
            embSize = embedding.shape[1]
            self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.posEmb = nn.Parameter(torch.tensor([[np.sin(pos/10000**(i/embSize)) if i%2==0 else np.cos(pos/10000**((i-1)/embSize)) for i in range(embSize)] for pos in range(seqMaxLen)], dtype=torch.float32), requires_grad=False) # seqLen × feaSize

        self.linformer = TransformerLayers_Linformer(seqMaxLen, pseTknNum, layersNum=L, feaSize=embSize, dk=H//A, multiNum=A, 
                                                     hdnDropout=hdnDropout)
        self.fcLinear = nn.Sequential(
                            nn.Linear(embSize*2, fcSize),
                            nn.BatchNorm1d(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = x + self.posEmb[:x.shape[1]]/3
        x,_,_,_,_ = self.linformer(x, data['maskPAD'].unsqueeze(1)&data['maskPAD'].unsqueeze(2)) # => batchSize × seqLen × embSize
        m = data['maskPAD'].int()

        x_max,_ = torch.max(x+(1-m).unsqueeze(dim=-1)*(-1e10), dim=1)
        x_mean = (x*m.unsqueeze(dim=-1)).sum(dim=1) / m.sum(dim=1).unsqueeze(dim=-1)
        # x,_ = torch.max(x, dim=1)

        return {'y_logit':self.fcLinear(torch.cat([x_max,x_mean], dim=-1))}

from linformer_pytorch import Linformer as pyLinformer

class Linformer(nn.Module):
    def __init__(self, classNum, seqMaxLen, 
                 embedding, 
                 pseTknNum=64,
                 L=6, H=512, A=8, fcSize=1024,
                 embDropout=0.2, hdnDropout=0.15):
        super(Linformer, self).__init__()
        

        if isinstance(embedding, VisioEmbedding):
            embSize = embedding.embedding[0].out_channels
            self.embedding = embedding
        elif isinstance(embedding, nn.Linear):
            embSize = embedding.weight.shape[0]
            self.embedding = embedding
        elif isinstance(embedding, InputCNN):
            embSize = H
            self.embedding = embedding
        else:
            embSize = embedding.shape[1]
            self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.posEmb = nn.Parameter(torch.tensor([[np.sin(pos/10000**(i/embSize)) if i%2==0 else np.cos(pos/10000**((i-1)/embSize)) for i in range(embSize)] for pos in range(seqMaxLen)], dtype=torch.float32), requires_grad=False) # seqLen × feaSize

        self.linformer = pyLinformer(input_size=seqMaxLen, channels=embSize, dim_d=H//A, dim_k=pseTknNum, dim_ff=H, dropout_ff=hdnDropout,
                                     nhead=A, depth=L, dropout=hdnDropout, 
                                     activation='gelu', checkpoint_level='C0', parameter_sharing='layerwise', 
                                     k_reduce_by_layer=0, full_attention=False, include_ff=True, w_o_intermediate_dim=None)
        self.fcLinear = nn.Sequential(
                            nn.Linear(embSize*2, fcSize),
                            nn.BatchNorm1d(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = x + self.posEmb[:x.shape[1]]/3
        x = self.linformer(x) # => batchSize × seqLen × embSize
        m = data['maskPAD'].int()

        x_max,_ = torch.max(x+(1-m).unsqueeze(dim=-1)*(-1e10), dim=1)
        x_mean = (x*m.unsqueeze(dim=-1)).sum(dim=1) / m.sum(dim=1).unsqueeze(dim=-1)
        # x,_ = torch.max(x, dim=1)

        return {'y_logit':self.fcLinear(torch.cat([x_max,x_mean], dim=-1))}

class cusBigBird(nn.Module):
    def __init__(self, classNum, seqMaxLen,
                 embedding, 
                 randomK=64,
                 L=6, H=512, A=8, fcSize=1024,
                 embDropout=0.2, hdnDropout=0.15):
        super(cusBigBird, self).__init__()
        

        if isinstance(embedding, VisioEmbedding):
            embSize = embedding.embedding[0].out_channels
            self.embedding = embedding
        elif isinstance(embedding, nn.Linear):
            embSize = embedding.weight.shape[0]
            self.embedding = embedding
        elif isinstance(embedding, InputCNN):
            embSize = H
            self.embedding = embedding
        else:
            embSize = embedding.shape[1]
            self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.posEmb = nn.Parameter(torch.tensor([[np.sin(pos/10000**(i/embSize)) if i%2==0 else np.cos(pos/10000**((i-1)/embSize)) for i in range(embSize)] for pos in range(seqMaxLen)], dtype=torch.float32), requires_grad=False) # seqLen × feaSize

        self.bigbird = TransformerLayers_BidBird(randomK, layersNum=L, feaSize=embSize, dk=H//A, multiNum=A, 
                                                 hdnDropout=hdnDropout)
        self.fcLinear = nn.Sequential(
                            nn.Linear(embSize*2, fcSize),
                            nn.BatchNorm1d(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = x + self.posEmb[:x.shape[1]]/3
        x,_,_,_ = self.bigbird(x, data['maskPAD'].unsqueeze(1)&data['maskPAD'].unsqueeze(2)) # => batchSize × seqLen × embSize
        m = data['maskPAD'].int()

        x_max,_ = torch.max(x+(1-m).unsqueeze(dim=-1)*(-1e10), dim=1)
        x_mean = (x*m.unsqueeze(dim=-1)).sum(dim=1) / m.sum(dim=1).unsqueeze(dim=-1)
        # x,_ = torch.max(x, dim=1)

        return {'y_logit':self.fcLinear(torch.cat([x_max,x_mean], dim=-1))}


class BigBird(nn.Module):
    def __init__(self, classNum, seqMaxLen,
                 embedding, 
                 L=6, H=512, A=8, fcSize=1024,
                 embDropout=0.2, hdnDropout=0.15):
        super(BigBird, self).__init__()

        if isinstance(embedding, VisioEmbedding):
            embSize = embedding.embedding[0].out_channels
            self.embedding = embedding
        elif isinstance(embedding, nn.Linear):
            embSize = embedding.weight.shape[0]
            self.embedding = embedding
        elif isinstance(embedding, InputCNN):
            embSize = H
            self.embedding = embedding
        else:
            embSize = embedding.shape[1]
            self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.posEmb = nn.Parameter(torch.tensor([[np.sin(pos/10000**(i/embSize)) if i%2==0 else np.cos(pos/10000**((i-1)/embSize)) for i in range(embSize)] for pos in range(seqMaxLen)], dtype=torch.float32), requires_grad=False) # seqLen × feaSize

        self.bigbird = TransformerLayers_BidBird(layersNum=L, feaSize=embSize, dk=H//A, multiNum=A, 
                                                  hdnDropout=hdnDropout)
        self.fcLinear = nn.Sequential(
                            nn.Linear(embSize*2, fcSize),
                            nn.BatchNorm1d(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = x + self.posEmb[:x.shape[1]]/3
        x = self.bigbird(x, data['maskPAD']) # => batchSize × seqLen × embSize
        m = data['maskPAD'].int()

        x_max,_ = torch.max(x+(1-m).unsqueeze(dim=-1)*(-1e10), dim=1)
        x_mean = (x*m.unsqueeze(dim=-1)).sum(dim=1) / m.sum(dim=1).unsqueeze(dim=-1)
        # x,_ = torch.max(x, dim=1)

        return {'y_logit':self.fcLinear(torch.cat([x_max,x_mean], dim=-1))}

class LSTransformer(nn.Module):
    def __init__(self, classNum, seqMaxLen,
                 embedding, r, w, 
                 L=6, H=512, A=8, fcSize=1024,
                 embDropout=0.2, hdnDropout=0.15):
        super(LSTransformer, self).__init__()
        self.w = w
        if isinstance(embedding, VisioEmbedding):
            embSize = embedding.embedding[0].out_channels
            self.embedding = embedding
        elif isinstance(embedding, nn.Linear):
            embSize = embedding.weight.shape[0]
            self.embedding = embedding
        elif isinstance(embedding, InputCNN):
            embSize = H
            self.embedding = embedding
        else:
            embSize = embedding.shape[1]
            self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.posEmb = nn.Parameter(torch.tensor([[np.sin(pos/10000**(i/embSize)) if i%2==0 else np.cos(pos/10000**((i-1)/embSize)) for i in range(embSize)] for pos in range(seqMaxLen)], dtype=torch.float32), requires_grad=False) # seqLen × feaSize

        self.lstransformer = TransforemrLayers_LS(r=r, w=w, layersNum=L, feaSize=embSize, dk=H//A, multiNum=A, 
                                                  hdnDropout=hdnDropout)
        self.fcLinear = nn.Sequential(
                            nn.Linear(embSize*2, fcSize),
                            nn.BatchNorm1d(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = x + self.posEmb[:x.shape[1]]/3
        x,_,_,_ = self.lstransformer(x, data['maskPAD']) # => batchSize × seqLen × embSize
        m = data['maskPAD'].int()

        x_max,_ = torch.max(x+(1-m).unsqueeze(dim=-1)*(-1e10), dim=1)
        x_mean = (x*m.unsqueeze(dim=-1)).sum(dim=1) / m.sum(dim=1).unsqueeze(dim=-1)
        # x,_ = torch.max(x, dim=1)

        return {'y_logit':self.fcLinear(torch.cat([x_max,x_mean], dim=-1))}

class FLASH(nn.Module):
    def __init__(self, classNum, seqMaxLen,
                 embedding, 
                 L=6, H=512, A=8, fcSize=1024,
                 embDropout=0.2, hdnDropout=0.15):
        super(FLASH, self).__init__()
        if isinstance(embedding, VisioEmbedding):
            embSize = embedding.embedding[0].out_channels
            self.embedding = embedding
        elif isinstance(embedding, nn.Linear):
            embSize = embedding.weight.shape[0]
            self.embedding = embedding
        elif isinstance(embedding, InputCNN):
            embSize = H
            self.embedding = embedding
        else:
            embSize = embedding.shape[1]
            self.embedding = TextEmbedding(torch.tensor(embedding, dtype=torch.float32), embDropout=embDropout, name='embedding')

        self.posEmb = nn.Parameter(torch.tensor([[np.sin(pos/10000**(i/embSize)) if i%2==0 else np.cos(pos/10000**((i-1)/embSize)) for i in range(embSize)] for pos in range(seqMaxLen)], dtype=torch.float32), requires_grad=False) # seqLen × feaSize

        self.flash = Transformer_FLASH(layersNum=L, feaSize=embSize, dk=H//A, multiNum=A, 
                                       hdnDropout=hdnDropout)
        self.fcLinear = nn.Sequential(
                            nn.Linear(embSize*2, fcSize),
                            nn.BatchNorm1d(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )

    def forward(self, data):
        x = self.embedding(data['tokenizedSeqArr']) # => batchSize × seqLen × embSize
        x = x + self.posEmb[:x.shape[1]]/3
        x,_,_,_ = self.flash(x, data['maskPAD']) # => batchSize × seqLen × embSize
        m = data['maskPAD'].int()

        x_max,_ = torch.max(x+(1-m).unsqueeze(dim=-1)*(-1e10), dim=1)
        x_mean = (x*m.unsqueeze(dim=-1)).sum(dim=1) / m.sum(dim=1).unsqueeze(dim=-1)
        # x,_ = torch.max(x, dim=1)

        return {'y_logit':self.fcLinear(torch.cat([x_max,x_mean], dim=-1))}

import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, classNum, pretrained=True, wh=32):
        super(ResNet18, self).__init__()
        self.net = models.resnet18(pretrained=pretrained)
        tmp = self.net.fc
        self.net.fc = nn.Linear(tmp.in_features, classNum)

        self.wh = wh
    def forward(self, data):
        x = data['tokenizedSeqArr'].transpose(-1,-2).reshape(-1,3,self.wh,self.wh)
        return {'y_logit':self.net(x)}

from transformers import AutoTokenizer, AutoModel
class RoBERTa(nn.Module):
    def __init__(self, classNum, weightPath='./pretrained_models/roberta-base', 
                 embSize=768, fcSize=1024, hdnDropout=0.1):
        super(RoBERTa, self).__init__()
        self.roberta = AutoModel.from_pretrained(weightPath)
        self.fcLinear = nn.Sequential(
                            nn.Linear(embSize*2, fcSize),
                            nn.BatchNorm1d(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )
    def forward(self, data):
        x = self.roberta(**data['inputData'])['last_hidden_state'] # => batchSize × seqLen × embSize
        m = data['inputData']['attention_mask']

        x_max,_ = torch.max(x+(1-m).unsqueeze(dim=-1)*(-1e10), dim=1)
        x_mean = (x*m.unsqueeze(dim=-1)).sum(dim=1) / m.sum(dim=1).unsqueeze(dim=-1)

        return {'y_logit':self.fcLinear(torch.cat([x_max,x_mean], dim=-1))}


# for generation

class Seq2SeqLanguageModel(BaseClassifier):
    def __init__(self, model, criterion=None, collateFunc=None, ignoreIdx=-100, 
                 AMP=False, DDP=False, FGM=False):
        self.model = model
        self.collateFunc = collateFunc
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignoreIdx) if criterion is None else criterion
        self.AMP,self.DDP,self.FGM = AMP,DDP,FGM
        if AMP:
            self.scaler = GradScaler()
    def calculate_y_logit(self, data, predict=False):
        if self.AMP:
            with autocast():
                return self.model(data, predict)
        else:
            return self.model(data, predict)
    def calculate_y_prob(self, data):
        Y_pre = self.calculate_y_logit(data, predict=True)['y_logit']
        return {'y_prob':F.softmax(Y_pre, dim=-1)}
    def calculate_y(self, data):
        Y_pre = self.calculate_y_logit(data, predict=True)['y_logit']
        return {'y_pre':torch.argmax(Y_pre, dim=-1)}
    def generate(self, data, beamwidth=4):
        if self.AMP:
            with autocast():
                res = self.model.beamsearch(data, beamwidth=beamwidth)
        else:
            res = self.model.beamsearch(data, beamwidth=beamwidth)
        res['y_pre'] = res['y_pre']
        sortIdx = res['scoreArr'].argsort(dim=1, descending=True)
        return {'y_pre':res['y_pre'][torch.arange(len(sortIdx)).unsqueeze(dim=-1),sortIdx], 'scoreArr':res['scoreArr'][torch.arange(len(sortIdx)).unsqueeze(dim=-1),sortIdx]}
    def calculate_loss_by_iterator(self, dataStream):
        loss,cnt = 0,0
        for data in dataStream:
            loss += self.calculate_loss(data) * len(data['tokenizedSeqArr'])
            cnt += len(data['tokenizedSeqArr'])
        return loss / cnt
    def calculate_y_prob_by_iterator(self, dataStream):
        device = next(self.model.parameters()).device
        YArr,Y_probArr,maskIdxArr = [],[],[]
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            Y_prob,Y,maskIdx = self.calculate_y_prob(data)['y_prob'][:,:-1].detach().cpu().data.numpy(),data['targetTknArr'][:,1:].detach().cpu().data.numpy(),data['tmaskPAD'][:,1:].detach().cpu().data.numpy()
            YArr.append(Y)
            Y_probArr.append(Y_prob)
            maskIdxArr.append(maskIdx)
        YArr,Y_probArr,maskIdxArr = np.vstack(YArr).astype('int32'),np.vstack(Y_probArr).astype('float32'),np.vstack(maskIdxArr).astype('bool')
        return {'y_prob':Y_probArr, 'y_true':YArr, 'mask_idx':maskIdxArr}
    
    def calculate_y_by_iterator(self, dataStream):
        device = next(self.model.parameters()).device
        YArr,Y_preArr,maskIdxArr = [],[],[]
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            Y_pre,Y,maskIdx = self.calculate_y(data)['y_pre'][:,:-1].detach().cpu().data.numpy(),data['targetTknArr'][:,1:].detach().cpu().data.numpy(),data['tMaskPAD'][:,1:].detach().cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
            maskIdxArr.append(maskIdx)
        YArr,Y_preArr,maskIdxArr = np.vstack(YArr).astype('int32'),np.vstack(Y_preArr).astype('int32'),np.vstack(maskIdxArr).astype('bool')
        return {'y_pre':Y_preArr, 'y_true':YArr, 'mask_idx':maskIdxArr}
    def calculate_metrics_by_iterator(self, dataStream, metrictor, ignoreIdx, report):
        # if self.collateFunc is not None:
        #     self.collateFunc.train = True
        device = next(self.model.parameters()).device
        YArr,Y_preArr,maskIdxArr = [],[],[]
        res,cnt = {},0
        for data in tqdm(dataStream):
            data = dict_to_device(data, device=device)
            Y_pre,Y,maskIdx = self.calculate_y(data)['y_pre'][:,:-1].detach().cpu().data.numpy().astype('int32'),data['targetTknArr'][:,1:].detach().cpu().data.numpy().astype('int32'),data['tMaskPAD'][:,1:].detach().cpu().data.numpy().astype('bool')
            batchData = {'y_pre':Y_pre, 'y_true':Y, 'mask_idx':maskIdx}
            metrictor.set_data(batchData, ignore_index=ignoreIdx)
            batchRes = metrictor(report, isPrint=False)
            for k in batchRes:
                res.setdefault(k, 0)
                res[k] += batchRes[k]*len(Y_pre)
            cnt += len(Y_pre)
        return {k:res[k]/cnt for k in res}
    def calculate_loss(self, data):
        out = self.calculate_y_logit(data)
        Y = data['targetTknArr'][:,1:].reshape(-1)
        Y_logit = out['y_logit'][:,:-1].reshape(len(Y),-1)
        maskIdx = data['tMaskPAD'][:,1:].reshape(-1)

        return self.criterion(Y_logit[maskIdx], Y[maskIdx]) # [maskIdx]
    def _train_step(self, data, optimizer):
        optimizer.zero_grad()
        if self.AMP:
            with autocast():
                loss = self.calculate_loss(data)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss = self.calculate_loss(data)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            optimizer.step()
        return loss.detach().cpu().data.numpy()
    def save(self, path, epochs, bestMtc=None):
        if self.DDP:
            model = self.model.module.state_dict()
        else:
            model = self.model.state_dict()
        stateDict = {'epochs':epochs, 'bestMtc':bestMtc, 'model':model}
        torch.save(stateDict, path)
        print('Model saved in "%s".'%path)
    def load(self, path, map_location=None):
        parameters = torch.load(path, map_location=map_location)
        if self.DDP:
            self.model.module.load_state_dict(parameters['model'])
        else:
            self.model.load_state_dict(parameters['model'])
        print("%d epochs and %.3lf val Score 's model load finished."%(parameters['epochs'], parameters['bestMtc'] if parameters['bestMtc'] is not None else 0.00))

class SingularformerForSeq2Seq(nn.Module):
    def __init__(self, classNum, seqMaxLen, sEmbedding, tEmbedding, 
                 L=6, H=384, A=6, fcSize=1024, tknDropout=0.0, embDropout=0.2, hdnDropout=0.15, gama=0.001):
        super(SingularformerForSeq2Seq, self).__init__()        
        self.sEmbedding = TextEmbedding(torch.tensor(sEmbedding, dtype=torch.float32), tknDropout=tknDropout, embDropout=embDropout, name='sEmbedding')
        self.tEmbedding = TextEmbedding(torch.tensor(tEmbedding, dtype=torch.float32), tknDropout=tknDropout, embDropout=embDropout, name='tEmbedding')
        self.posEmb = nn.Parameter(torch.tensor([[np.sin(pos/10000**(i/H)) if i%2==0 else np.cos(pos/10000**((i-1)/H)) for i in range(H)] for pos in range(seqMaxLen)], dtype=torch.float32), requires_grad=False) # seqLen × feaSize
        assert sEmbedding.shape[1]==tEmbedding.shape[1]==H
        self.encoder = SingularformerEncoderLayers(L, H, H//A, A, r=H//A, hdnDropout=hdnDropout, name='encoder')
        self.decoder = SingularformerDecoderLayers(L, H, H//A, A, r=H//A, hdnDropout=hdnDropout, name='decoder')
        self.gama = gama
        self.fcLinear = nn.Sequential(
                            nn.Linear(H, fcSize),
                            nn.LayerNorm(fcSize),
                            nn.ReLU(),
                            nn.Dropout(hdnDropout),
                            nn.Linear(fcSize, classNum)
                        )
        self.classNum = classNum
        self.alwaysTrain = False
    def beamsearch(self, data, beamwidth=8):
        source = self.sEmbedding(data['sourceTknArr']) # => batchSize × sL × embSize
        B,sL,C = source.shape
        source = source+self.posEmb[:source.shape[1]]/3
        source,_,_,_,_,_ = self.encoder(source, data['sMaskPAD'].unsqueeze(1)&data['sMaskPAD'].unsqueeze(2)) # => batchSize × seqLen × embSize

        y_pre = torch.zeros((B,beamwidth,data['tMaxLen']), dtype=data['sourceTknArr'].dtype,device=source.device) # => B × bw × tL
        scoreArr = -torch.inf*torch.ones((B,beamwidth), dtype=source.dtype, device=source.device) # => B × bw

        locTarTknArr = data['targetTknArr'][:,[0]] # => B × 1
        isPredicting = torch.ones((B,beamwidth), dtype=torch.bool, device=source.device) # => B × bw
        locScoreArr = torch.zeros((B,beamwidth)) # => B × bw

        isFirst,cnt = True,0
        dMaskPAD1,dMaskPAD2 = torch.tril(torch.ones((B,data['tMaxLen'],data['tMaxLen']), dtype=torch.bool, device=source.device), 0),data['sMaskPAD'].unsqueeze(dim=1).repeat(1,data['tMaxLen'],1) # => batchSize × tL × sL
        while cnt<data['tMaxLen']:
            if not isFirst:
                locTarTknArr = locTarTknArr.reshape(B*beamwidth, -1) # => B*bw × *L
            target = self.tEmbedding(locTarTknArr) # => B × *L × embSize
            target = target+self.posEmb[source.shape[1]:source.shape[1]+target.shape[1]]/3

            target,_,_,_,_,_,_ = self.decoder(target, source, dMaskPAD1[:,:cnt+1,:cnt+1], dMaskPAD2[:,:cnt+1])

            y_logit_ = self.fcLinear(target[:,-1]) # => B(*bw) × classNum
                
            if isFirst:
                sortScore_,sortIdx_ = y_logit_.sort(dim=-1, descending=True) # => B × classNum
                locTarTknArr = torch.cat([locTarTknArr.unsqueeze(dim=1).repeat(1,beamwidth,1),sortIdx_[:,:beamwidth].unsqueeze(dim=2)], dim=2) # => B × bw × 2
                
                locScoreArr = F.log_softmax(sortScore_, dim=-1)[:,:beamwidth] # B × bw

                source = source.unsqueeze(1).repeat(1,beamwidth,1,1).reshape(-1, sL, C) # => B*bw × sL × C
                isFirst = False

                dMaskPAD1,dMaskPAD2 = dMaskPAD1.unsqueeze(dim=1).repeat(1,beamwidth,1,1).reshape(B*beamwidth, data['tMaxLen'],data['tMaxLen']),dMaskPAD2.unsqueeze(dim=1).repeat(1,beamwidth,1,1).reshape(B*beamwidth, data['tMaxLen'],sL)
            else:
                y_logit_ = y_logit_.reshape(B,beamwidth,-1) # => B × bw × classNum
                sortScore_,sortIdx_ = y_logit_.sort(dim=-1, descending=True) # => B × bw × classNum

                locScoreArr_ = (locScoreArr.unsqueeze(dim=2)*cnt + F.log_softmax(sortScore_, dim=-1)[:,:,:beamwidth])/(cnt+1) # => B × bw × bw
                locScoreArr_[~isPredicting] = -torch.inf # for the results with [EOS], set their score to -inf, locScoreArr[~isPredicting].reshape(-1,1)

                topScore,topIdx = locScoreArr_.reshape(B,-1).sort(dim=1, descending=True) # => B × bw*bw
                topScore,topIdx = topScore[:,:beamwidth],topIdx[:,:beamwidth] # => B × bw

                locScoreArr = topScore
                locTarTknArr = locTarTknArr.reshape(B,beamwidth,-1) # => B × bw × *L

                reIdx = topIdx//beamwidth # => B × bw
                isPredicting = isPredicting[torch.arange(B).unsqueeze(-1), reIdx] # => B × bw
                locTarTknArr = locTarTknArr[torch.arange(B).unsqueeze(-1),reIdx] # => B × bw × *L
                locIdx = sortIdx_[:,:,:beamwidth].reshape(B,-1)[torch.arange(B).unsqueeze(-1),topIdx] # => B × bw 
                locTarTknArr = torch.cat([locTarTknArr,locIdx.unsqueeze(dim=2)], dim=2) # => B × bw × *L

            newi,newj = torch.where(isPredicting&(locTarTknArr[:,:,-1]<=1))
            if len(newi)>0: # has some new reult with [EOS], put them into y_pre 
                for i,j in zip(newi,newj):
                    k = scoreArr[i].argmin()
                    if locScoreArr[i,j]>scoreArr[i,k]:
                        y_pre[i,k,:cnt+1] = locTarTknArr[i,j,1:cnt+2]
                        scoreArr[i,k] = locScoreArr[i,j]

            isPredicting = isPredicting&(locTarTknArr[:,:,-1]>1) # => B × bw

            cnt += 1
            if torch.sum(isPredicting)==0:
                break

        return {'y_pre':y_pre, 'scoreArr':scoreArr}
    def forward(self, data, predict=False):
        source = self.sEmbedding(data['sourceTknArr']) # => batchSize × seqLen × embSize
        B,sL,C = source.shape
        source = source+self.posEmb[:source.shape[1]]/3
        addLoss = 0
        source,_,_,_,_,addLoss_ = self.encoder(source, data['sMaskPAD'].unsqueeze(1)&data['sMaskPAD'].unsqueeze(2)) # => batchSize × seqLen × embSize
        if predict and not self.alwaysTrain:
            y_logit = None
            locTarTknArr = data['targetTknArr'][:,[0]] # => B × 1
            isPredicting = torch.ones(B, dtype=torch.bool, device=source.device)
            
            cnt = 0
            dMaskPAD1,dMaskPAD2 = torch.tril(torch.ones((B,data['tMaxLen'],data['tMaxLen']), dtype=torch.bool, device=source.device), 0),data['sMaskPAD'].unsqueeze(dim=1).repeat(1,data['tMaxLen'],1) # => batchSize × tL × sL
            while isPredicting.sum()>0 and cnt<data['tMaxLen']:
                target = self.tEmbedding(locTarTknArr[isPredicting]) # => batchSize × seqLen × embSize
                target = target+self.posEmb[source.shape[1]:source.shape[1]+target.shape[1]]/3

                target,_,_,_,_,_,_ = self.decoder(target, source[isPredicting], dMaskPAD1[isPredicting,:cnt+1,:cnt+1], dMaskPAD2[isPredicting,:cnt+1])
                y_logit_ = self.fcLinear(target[:,-1])
                
                if y_logit is None:
                    y_logit = torch.zeros((B, data['tMaxLen'], self.classNum), dtype=y_logit_.dtype, device=y_logit_.device)
                y_logit[isPredicting, cnt] = y_logit_

                locTarTknArr = torch.cat([locTarTknArr, torch.argmax(y_logit[:,cnt], dim=-1, keepdims=True)], dim=1) # => batchSize × *L
                isPredicting = isPredicting&(locTarTknArr[:,-1]>1)

                cnt += 1
        else:
            addLoss += addLoss_
            target = self.tEmbedding(data['targetTknArr']) # => batchSize × seqLen × embSize
            target = target+self.posEmb[source.shape[1]:source.shape[1]+target.shape[1]]/3

            target,_,_,_,_,_,addLoss = self.decoder(target, source, data['dMaskPAD1'], data['dMaskPAD2'])
            addLoss += addLoss_

            y_logit = self.fcLinear(target)
        return {'y_logit':y_logit, 'add_loss':addLoss*self.gama/2}