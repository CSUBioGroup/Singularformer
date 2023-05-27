from torch import nn as nn
from torch.nn import functional as F
import torch,time,os,random
import numpy as np
from collections import OrderedDict
import torchvision

class nnTranspose(nn.Module):
    def __init__(self, dim1=-1, dim2=-2):
        super(nnTranspose, self).__init__()
        self.dim1,self.dim2 = dim1,dim2
    def forward(self, x):
        return x.transpose(self.dim1,self.dim2)

class TextEmbedding(nn.Module):
    def __init__(self, embedding, embDropout=0.3, freeze=False, name='textEmbedding'):
        super(TextEmbedding, self).__init__()
        self.name = name
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding,dtype=torch.float32), freeze=freeze)
        self.dropout = nn.Dropout(p=embDropout)
    def forward(self, x):
        # x: batchSize × seqLen
        x = self.dropout(self.embedding(x))
        return x

class VisioEmbedding(nn.Module):
    def __init__(self, embDropout=0.3, wh=32, name='visioEmbedding'):
        super(VisioEmbedding, self).__init__()
        tmp = torchvision.models.resnet18(pretrained=True)
        # tmp.conv1.stride = (1,1)
        self.embedding = nn.Sequential(tmp.conv1, tmp.bn1, tmp.relu) # , tmp.maxpool
        self.dropout = nn.Dropout(p=embDropout)
        self.wh = wh
    def forward(self, x):
        # x: batchSize × w*h × 3
        x = self.dropout(self.embedding(x.transpose(-1,-2).reshape(-1,3,self.wh,self.wh)))
        return x.reshape(x.shape[0],x.shape[1],-1).transpose(-1,-2)

class SelfAttention_PreLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, name='selfAttn'):
        super(SelfAttention_PreLN, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.layerNorm1 = nn.LayerNorm([feaSize])
        self.WQ = nn.Linear(feaSize, self.dk*multiNum)
        self.WK = nn.Linear(feaSize, self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.name = name
    def forward(self, qx,kx,vx, maskPAD=None):
        # x: batchSize × seqLen × feaSize; 
        B,L,C = qx.shape
        qx,kx,vx = self.layerNorm1(qx),self.layerNorm1(kx),self.layerNorm1(vx)
        queries = self.WQ(qx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        keys    = self.WK(kx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        values  = self.WV(vx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
    
        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × seqLen × seqLen

        if maskPAD is not None:
            scores = scores.masked_fill((maskPAD==0).unsqueeze(dim=1), -2**15+1) # -np.inf

        alpha = F.softmax(scores, dim=3)

        z = alpha @ values # => batchSize × multiNum × seqLen × dk
        z = z.transpose(1,2).reshape(B,L,-1) # => batchSize × seqLen × multiNum*dk

        z = self.WO(z) # => batchSize × seqLen × feaSize
        return z

class FFN_PreLN(nn.Module):
    def __init__(self, feaSize, dropout=0.1, name='FFN'):
        super(FFN_PreLN, self).__init__()
        
        self.layerNorm2 = nn.LayerNorm([feaSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(feaSize, feaSize*4), 
                        nn.GeLU(), 
                        nn.Linear(feaSize*4, feaSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = x + self.dropout(z) # => batchSize × seqLen × feaSize
        ffnx = self.Wffn(self.layerNorm2(z)) # => batchSize × seqLen × feaSize
        return z+self.dropout(ffnx) # => batchSize × seqLen × feaSize
    
class Transformer_PreLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1):
        super(Transformer_PreLN, self).__init__()
        self.selfAttn = SelfAttention_PreLN(feaSize, dk, multiNum)
        self.ffn = FFN_PreLN(feaSize, dropout)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx, maskPAD = input
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        z = self.selfAttn(qx,kx,vx, maskPAD) # => batchSize × seqLen × feaSize
        x = self.ffn(vx, z)
        return (x,x,x, maskPAD) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class TransformerLayers_PreLN(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, dropout=0.1, usePos=True, name='textTransformer'):
        super(TransformerLayers_PreLN, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_PreLN(feaSize, dk, multiNum, dropout)) for i in range(layersNum)]
                                     )
                                 )
        self.name = name
        self.usePos = usePos
    def forward(self, x, maskPAD=None):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,maskPAD = self.transformerLayers((x,x,x, maskPAD))
        return (qx,kx,vx,maskPAD) # => batchSize × seqLen × feaSize

class SelfAttention_PostLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, name='selfAttn'):
        super(SelfAttention_PostLN, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.Linear(feaSize, self.dk*multiNum)
        self.WK = nn.Linear(feaSize, self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.name = name
    def forward(self, qx, kx, vx, maskPAD=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        B,L,C = qx.shape
        queries = self.WQ(qx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        keys    = self.WK(kx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        values  = self.WV(vx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
    
        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × seqLen × seqLen

        if maskPAD is not None:
            scores = scores.masked_fill((maskPAD==0).unsqueeze(dim=1), -2**15+1) # -np.inf

        alpha = F.softmax(scores, dim=3)

        z = alpha @ values # => batchSize × multiNum × seqLen × dk
        z = z.transpose(1,2).reshape(B,L,-1) # => batchSize × seqLen × multiNum*dk

        z = self.WO(z) # => batchSize × seqLen × feaSize
        return z

class FFN_PostLN(nn.Module):
    def __init__(self, feaSize, dropout=0.1, name='FFN'):
        super(FFN_PostLN, self).__init__()
        self.layerNorm1 = nn.LayerNorm([feaSize])
        self.layerNorm2 = nn.LayerNorm([feaSize])
        self.Wffn = nn.Sequential(
                        nn.Linear(feaSize, feaSize*4), 
                        nn.GeLU(),
                        nn.Linear(feaSize*4, feaSize)
                    )
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x, z):
        z = self.layerNorm1(x + self.dropout(z)) # => batchSize × seqLen × feaSize
        ffnx = self.Wffn(z) # => batchSize × seqLen × feaSize

        return self.layerNorm2(z+self.dropout(ffnx)) # => batchSize × seqLen × feaSize
    
class Transformer_PostLN(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1):
        super(Transformer_PostLN, self).__init__()
        self.selfAttn = SelfAttention_PostLN(feaSize, dk, multiNum)
        self.ffn = FFN_PostLN(feaSize, dropout)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx, maskPAD = input
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        z = self.selfAttn(qx,kx,vx, maskPAD) # => batchSize × seqLen × feaSize
        x = self.ffn(vx, z)
        return (x,x,x,maskPAD) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class TransformerLayers_PostLN(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, dropout=0.1, usePos=True, name='textTransformer'):
        super(TransformerLayers_PostLN, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_PostLN(feaSize, dk, multiNum, dropout)) for i in range(layersNum)]
                                     )
                                 )
        self.name = name
        self.usePos = usePos
    def forward(self, x, maskPAD=None):
        # x: batchSize × seqLen × feaSize;
        qx,kx,vx,maskPAD = self.transformerLayers((x,x,x, maskPAD))
        return  (qx,kx,vx,maskPAD) # => batchSize × seqLen × feaSize

class SelfAttention_Realformer(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1, name='selfAttn'): # maxRelativeDist=7, 
        super(SelfAttention_Realformer, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.Linear(feaSize, self.dk*multiNum)
        self.WK = nn.Linear(feaSize, self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.dropout = nn.Dropout(p=dropout)
        # if maxRelativeDist>0:
        #     self.relativePosEmbK = nn.Embedding(2*maxRelativeDist+1, multiNum)
        #     self.relativePosEmbB = nn.Embedding(2*maxRelativeDist+1, multiNum)
        # self.maxRelativeDist = maxRelativeDist
        self.name = name
    def forward(self, qx, kx, vx, preScores=None, maskPAD=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        B,L,C = qx.shape
        queries = self.WQ(qx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        keys    = self.WK(kx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        values  = self.WV(vx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
    
        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × seqLen × seqLen
        # if preScores is None:
        #     print(scores.shape)
        #     print(scores.sum(dim=-1).sum(dim=-1).sum(dim=-1))
        #     print(scores[0].sum(dim=-1).sum(dim=-1).sum(dim=-1))
        #     print(scores.sum(dim=-1).sum(dim=-1).sum(dim=-1)[0])  
        # relative position embedding
        # if self.maxRelativeDist>0:
        #     relativePosTab = torch.abs(torch.arange(0,L).reshape(-1,1) - torch.arange(0,L).reshape(1,-1)).float() # L × L
        #     relativePosTab[relativePosTab>self.maxRelativeDist] = self.maxRelativeDist+torch.log2(relativePosTab[relativePosTab>self.maxRelativeDist]-self.maxRelativeDist).float()
        #     relativePosTab = torch.clip(relativePosTab,min=0,max=self.maxRelativeDist*2).long().to(qx.device)
        #     scores = scores * self.relativePosEmbK(relativePosTab).transpose(0,-1).reshape(1,self.multiNum,L,L) + self.relativePosEmbB(relativePosTab).transpose(0,-1).reshape(1,self.multiNum,L,L)

        # residual attention
        if preScores is not None:
            scores = scores + preScores

        if maskPAD is not None:
            #scores = scores*maskPAD.unsqueeze(dim=1)
            scores = scores.masked_fill((maskPAD==0).unsqueeze(dim=1), -2**15+1) # -np.inf

        alpha = self.dropout(F.softmax(scores, dim=3))

        z = alpha @ values # => batchSize × multiNum × seqLen × dk
        z = z.transpose(1,2).reshape(B,L,-1) # => batchSize × seqLen × multiNum*dk

        z = self.WO(z) # => batchSize × seqLen × feaSize
        return z,scores
    
class Transformer_Realformer(nn.Module):
    def __init__(self, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.GELU):
        super(Transformer_Realformer, self).__init__()
        self.selfAttn = SelfAttention_Realformer(feaSize, dk, multiNum, dropout)
        self.ffn = FFN_PostLN(feaSize, dropout)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx,preScores,maskPAD = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z,preScores = self.selfAttn(qx,kx,vx,preScores,maskPAD) # => batchSize × seqLen × feaSize
        x = self.ffn(vx, z)
        return (x, x, x, preScores,maskPAD) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class TransformerLayers_Realformer(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, hdnDropout=0.1, 
                 actFunc=nn.GELU, name='textTransformer'):
        super(TransformerLayers_Realformer, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_Realformer(feaSize, dk, multiNum, hdnDropout, actFunc)) for i in range(layersNum)]
                                     )
                                 )
        self.name = name
    def forward(self, x, maskPAD):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD = self.transformerLayers((x, x, x, None, maskPAD))
        return (qx,kx,vx,scores,maskPAD)# => batchSize × seqLen × feaSize


class SelfAttention_Linformer(nn.Module):
    def __init__(self, feaSize, dk, multiNum, seqMaxLen, pseudoTknNum, dropout=0.1, name='selfAttn'): # maxRelativeDist=7, 
        super(SelfAttention_Linformer, self).__init__()
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.Linear(feaSize, self.dk*multiNum)
        self.WK = nn.Linear(feaSize, self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)

        self.E = nn.Linear(seqMaxLen, pseudoTknNum)
        self.F = nn.Linear(seqMaxLen, pseudoTknNum)

        self.dropout = nn.Dropout(p=dropout)
        # if maxRelativeDist>0:
        #     self.relativePosEmbK = nn.Embedding(2*maxRelativeDist+1, multiNum)
        #     self.relativePosEmbB = nn.Embedding(2*maxRelativeDist+1, multiNum)
        # self.maxRelativeDist = maxRelativeDist
        self.pseudoTknNum = pseudoTknNum
        self.name = name
    def forward(self, qx, kx, vx, preScores=None, maskPAD=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        B,L,C = qx.shape
        queries = self.WQ(qx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        
        kx = self.E(kx.transpose(-1,-2)).transpose(-1,-2) # => batchSize × pseudoTknNum × feaSize
        keys = self.WK(kx).reshape(B,self.pseudoTknNum,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × pseudoTknNum × dk

        vx = self.F(vx.transpose(-1,-2)).transpose(-1,-2)
        values  = self.WV(vx).reshape(B,self.pseudoTknNum,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × pseudoTknNum × dk

        # keys = self.WK(kx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        # keys = self.E(keys.transpose(-1,-2)).transpose(-1,-2) # => batchSize × multiNum × pseudoTknNum × dk

        # values = self.WK(vx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        # values = self.F(values.transpose(-1,-2)).transpose(-1,-2) # => batchSize × multiNum × pseudoTknNum × dk

        scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × seqLen × pseudoTknNum
        # if preScores is None:
        #     print(scores.shape)
        #     print(scores.sum(dim=-1).sum(dim=-1).sum(dim=-1))
        #     print(scores[0].sum(dim=-1).sum(dim=-1).sum(dim=-1))
        #     print(scores.sum(dim=-1).sum(dim=-1).sum(dim=-1)[0])  
        # relative position embedding
        # if self.maxRelativeDist>0:
        #     relativePosTab = torch.abs(torch.arange(0,L).reshape(-1,1) - torch.arange(0,L).reshape(1,-1)).float() # L × L
        #     relativePosTab[relativePosTab>self.maxRelativeDist] = self.maxRelativeDist+torch.log2(relativePosTab[relativePosTab>self.maxRelativeDist]-self.maxRelativeDist).float()
        #     relativePosTab = torch.clip(relativePosTab,min=0,max=self.maxRelativeDist*2).long().to(qx.device)
        #     scores = scores * self.relativePosEmbK(relativePosTab).transpose(0,-1).reshape(1,self.multiNum,L,L) + self.relativePosEmbB(relativePosTab).transpose(0,-1).reshape(1,self.multiNum,L,L)

        # residual attention
        if preScores is not None:
            scores = scores + preScores

        if maskPAD is not None:
            scores = scores.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=1).unsqueeze(dim=-1), -2**15+1) # => batchSize × multiNum × seqLen × pseudoTknNum

        alpha = self.dropout(F.softmax(scores, dim=3))

        z = alpha @ values # => batchSize × multiNum × seqLen × dk
        z = z.transpose(1,2).reshape(B,L,-1) # => batchSize × seqLen × multiNum*dk

        z = self.WO(z) # => batchSize × seqLen × feaSize
        return z,scores

class Transformer_Linformer(nn.Module):
    def __init__(self, seqMaxLen, pseudoTknNum, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.GELU):
        super(Transformer_Linformer, self).__init__()
        self.selfAttn = SelfAttention_Linformer(feaSize, dk, multiNum, seqMaxLen, pseudoTknNum, dropout)
        self.ffn = FFN_PostLN(feaSize, dropout, actFunc)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx,preScores,maskPAD = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z,preScores = self.selfAttn(qx,kx,vx,preScores,maskPAD) # => batchSize × seqLen × feaSize
        x = self.ffn(vx, z)
        return (x, x, x, preScores,maskPAD) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class TransformerLayers_Linformer(nn.Module):
    def __init__(self, seqMaxLen, pseudoTknNum, layersNum, feaSize, dk, multiNum, hdnDropout=0.1, 
                 actFunc=nn.GELU, name='textTransformer'):
        super(TransformerLayers_Linformer, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_Linformer(seqMaxLen, pseudoTknNum, feaSize, dk, multiNum, hdnDropout, actFunc)) for i in range(layersNum)]
                                     )
                                 )
        self.name = name
    def forward(self, x, maskPAD):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD = self.transformerLayers((x, x, x, None, maskPAD))
        return (qx,kx,vx,scores,maskPAD)# => batchSize × seqLen × feaSize

class SelfAttention_BigBird(nn.Module):
    def __init__(self, feaSize, dk, multiNum, randomK, name='selfAttn'):
        super(SelfAttention_BigBird, self).__init__()
        self.randomK = randomK
        self.dk = dk
        self.multiNum = multiNum
        self.WQ = nn.Linear(feaSize, self.dk*multiNum)
        self.WK = nn.Linear(feaSize, self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.name = name
    def forward(self, qx, kx, vx, maskPAD=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        B,L,C = qx.shape
        queries = self.WQ(qx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        keys    = self.WK(kx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
        values  = self.WV(vx).reshape(B,L,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × seqLen × dk
    
        scores = torch.zeros((B,self.multiNum,L,L), dtype=queries.dtype, device=queries.device) # => batchSize × multiNum × seqLen × seqLen
        # Global Attention
        # 1st & last token attends all other tokens
        scores[:,:,[0,-1],:] = queries[:,:,[0,-1],:] @ keys.transpose(-1,-2) # => batchSize × multiNum × 1 × seqLen
        # 1st & last token getting attended by all other tokens
        scores[:,:,:,[0,-1]] = queries @ keys[:,:,[0,-1],:].transpose(-1,-2) # => batchSize × multiNum × seqLen × 1

        # Sliding Attention
        scores[:,:,range(0,L-1),range(1,L)] = (queries[:,:,:-1,:].unsqueeze(3) @ keys[:,:,list(range(1,L)),:].unsqueeze(-1)).squeeze() # => batchSize × multiNum × seqLen-1
        scores[:,:,range(1,L),range(0,L-1)] = (queries[:,:,1:,:].unsqueeze(3) @ keys[:,:,list(range(0,L-1)),:].unsqueeze(-1)).squeeze() # => batchSize × multiNum × seqLen-1
        scores[:,:,range(L),range(L)] = (queries.unsqueeze(3) @ keys.unsqueeze(-1)).squeeze() # => batchSize × multiNum × seqLen

        # Random Attention
        r = range(1,L-1, (L-2)//self.randomK)
        scores[:,:,r,:][:,:,:,r] = queries[:,:,r] @ keys[:,:,r].transpose(-1,-2) # => batchSize × multiNum × randomK × randomK

        scores /= np.sqrt(self.dk)

        if maskPAD is not None:
            scores = scores.masked_fill((maskPAD==0).unsqueeze(dim=1), -2**15+1) # -np.inf

        alpha = F.softmax(scores, dim=3)

        z = alpha @ values # => batchSize × multiNum × seqLen × dk
        z = z.transpose(1,2).reshape(B,L,-1) # => batchSize × seqLen × multiNum*dk

        z = self.WO(z) # => batchSize × seqLen × feaSize
        return z
    
class Transformer_BidBird(nn.Module):
    def __init__(self, randomK, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.GELU):
        super(Transformer_BidBird, self).__init__()
        self.selfAttn = SelfAttention_BigBird(feaSize, dk, multiNum, randomK)
        self.ffn = FFN_Realformer(feaSize, dropout, actFunc)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx, maskPAD = input
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        z = self.selfAttn(qx,kx,vx, maskPAD) # => batchSize × seqLen × feaSize
        x = self.ffn(vx, z)
        return (x,x,x,maskPAD) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class TransformerLayers_cusBidBird(nn.Module):
    def __init__(self, randomK, layersNum, feaSize, dk, multiNum, hdnDropout=0.1, usePos=True, name='textTransformer'):
        super(TransformerLayers_cusBidBird, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, Transformer_BidBird(randomK, feaSize, dk, multiNum, hdnDropout)) for i in range(layersNum)]
                                     )
                                 )
        self.name = name
        self.usePos = usePos
    def forward(self, x, maskPAD=None):
        # x: batchSize × seqLen × feaSize;
        qx,kx,vx,maskPAD = self.transformerLayers((x,x,x, maskPAD))
        return  (qx,kx,vx,maskPAD) # => batchSize × seqLen × feaSize

from pyBigBird import utils as bbutils
from pyBigBird.layers import EncoderLayer as pybigbird
class TransformerLayers_BidBird(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, hdnDropout=0.1, usePos=True, name='textTransformer'):
        super(TransformerLayers_BidBird, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, pybigbird(feaSize, feaSize*4, multiNum, hdnDropout)) for i in range(layersNum)]
                                     )
                                 )
        self.name = name
        self.usePos = usePos
    def forward(self, x, maskPAD=None):
        # x: batchSize × seqLen × feaSize;
        batch_size,encoder_length,emb_size = x.shape
    
        input_blocked_mask = maskPAD.view(batch_size, encoder_length // 16, 16)
        from_mask = maskPAD.view(batch_size, 1, encoder_length, 1)
        to_mask = maskPAD.view(batch_size, 1, 1, encoder_length)
        band_mask = bbutils.create_band_mask_from_inputs(input_blocked_mask, input_blocked_mask)

        x,attention_mask,band_mask,from_mask,to_mask,input_blocked_mask = self.transformerLayers((x,None,band_mask,from_mask,to_mask,input_blocked_mask))
        return  x # => batchSize × seqLen × feaSize

from pyLSTransformer.model import Transformer as LSTransfoirmer
import pandas as pd
class pyLS(nn.Module):
    def __init__(self, r, w, feaSize, dk, multiNum, hdnDropout=0.1):
        super(pyLS, self).__init__()
        config = pd.Series()
        config['transformer_dim'] = feaSize
        config['attn_type'] = 'lsta'
        config['pooling_mode'] = 'MEAN'
        config['num_head'] = multiNum
        config['head_dim'] = dk
        config['num_landmarks'] = r
        config['max_seq_len'] = 4096
        config['attention_dropout'] = hdnDropout
        config['window_size'] = w
        config['fp32_attn'] = False
        config['dropout_prob'] = hdnDropout
        config['debug'] = False
        config['transformer_hidden_dim'] = feaSize*4
        self.lsTransformer = LSTransfoirmer(config)
    def forward(self, input):
        qx,kx,vx,maskPAD = input
        x = self.lsTransformer(X=vx, mask=maskPAD)
        return (x, x, x, maskPAD) # => batchSize × seqLen × feaSize

class TransforemrLayers_LS(nn.Module):
    def __init__(self, r, w, layersNum, feaSize, dk, multiNum, hdnDropout=0.1, name='textTransformer'):
        super(TransforemrLayers_LS, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, pyLS(r, w, feaSize, dk, multiNum, hdnDropout)) for i in range(layersNum)]
                                     )
                                 )
        self.name = name
    def forward(self, x, maskPAD):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,maskPAD = self.transformerLayers((x, x, x, maskPAD.int()))
        return (qx,kx,vx,maskPAD)# => batchSize × seqLen × feaSize

from flash_pytorch import FLASH
class pyFLASH(nn.Module):
    def __init__(self, feaSize, dk, multiNum, hdnDropout=0.1):
        super(pyFLASH, self).__init__()
        self.flash = FLASH(dim=feaSize, query_key_dim=dk, 
                           dropout=hdnDropout,
                           expansion_factor=4.0,
                           group_size=256, laplace_attn_fn=True)
    def forward(self, input):
        qx,kx,vx,maskPAD = input
        x = self.flash(x=vx, mask=maskPAD)
        return (x,x,x,maskPAD)

class Transformer_FLASH(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, hdnDropout=0.1, name='textTransformer'):
        super(Transformer_FLASH, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, pyFLASH(feaSize, dk, multiNum, hdnDropout)) for i in range(layersNum)]
                                     )
                                 )
        self.name = name
    def forward(self, x, maskPAD):
        qx,kx,vx,maskPAD = self.transformerLayers((x,x,x,maskPAD))
        return (qx,kx,vx,maskPAD)# => batchSize × seqLen × feaSize

def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class SingularAttention(nn.Module):
    def __init__(self, r, feaSize, dk, multiNum, dropout=0.1, linearLevel='l1', name='selfAttn'):
        super(SingularAttention, self).__init__()
        self.r = r
        self.dk = dk
        self.multiNum = multiNum
        
        if linearLevel=='l1':
            self.pseudoAttn = nn.Linear(feaSize, r)
        elif linearLevel=='l2':
            self.pseudoAttn = nn.Linear(feaSize, multiNum*r)
        elif linearLevel=='l3':
            self.fQKV = nn.Linear(feaSize, r)
            self.fU = nn.Linear(feaSize, r)
        elif linearLevel=='l4':
            self.fQ = nn.Linear(feaSize, r)
            self.fK = nn.Linear(feaSize, r)
            self.fU = nn.Linear(feaSize, r)
            self.fV = nn.Linear(feaSize, r)
        elif linearLevel=='l5':
            self.fQ = nn.Linear(feaSize, multiNum*r)
            self.fK = nn.Linear(feaSize, multiNum*r)
            self.fU = nn.Linear(feaSize, multiNum*r)
            self.fV = nn.Linear(feaSize, multiNum*r)
        elif linearLevel=='l3':
            self.fKV = nn.Linear(feaSize, r)

        self.WQ = nn.Linear(feaSize, self.dk*multiNum)
        self.WK = nn.Linear(feaSize, self.dk*multiNum)
        self.WV = nn.Linear(feaSize, self.dk*multiNum)
        self.linearLevel = linearLevel

        self.WO = nn.Linear(self.dk*multiNum, feaSize)
        self.dropout = nn.Dropout(p=dropout)
        self.name = name

        # self.I = nn.Parameter(torch.eye(self.r, dtype=torch.float32).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.ones_minus_I = nn.Parameter((torch.ones((self.r,self.r), dtype=torch.bool) ^ torch.eye(self.r, dtype=torch.bool)).unsqueeze(0).unsqueeze(0), requires_grad=False)
        
    def forward(self, qx, kx, vx, addLoss, preScores=None, maskPAD=None):
        # x: batchSize × seqLen × feaSize; maskPAD: batchSize × seqLen × seqLen
        # obtain the pseudo tokens
        B,L,C = qx.shape

        if self.linearLevel=='l1':
            pScore = self.pseudoAttn(vx) # => batchSize × seqLen × r
            if maskPAD is not None:
                pScore = pScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)

            pAlpha = F.softmax(pScore, dim=-2).transpose(-1,-2) # => batchSize × r × seqLen
            pAlpha_ = F.softmax(pScore, dim=-1) # => batchSize × seqLen × r

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean((((pAlpha @ pAlpha.transpose(-1,-2))*self.ones_minus_I[0])**2))

            qx = self.dropout(pAlpha) @ qx # => batchSize × r × feaSize
            kx = self.dropout(pAlpha) @ kx # => batchSize × r × feaSize
            vx = self.dropout(pAlpha) @ vx # => batchSize × r × feaSize

            queries = self.WQ(qx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            keys    = self.WK(kx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            values  = self.WV(vx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
        
            scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × r × r

            # residual attention
            if preScores is not None:
                scores = scores + preScores

            alpha = F.softmax(scores, dim=3) # batchSize × multiNum × r × r

            if qx.requires_grad:
                # add the diagonal loss
                addLoss += torch.mean(alpha * self.ones_minus_I)
                # addLoss += torch.mean((alpha * self.ones_minus_I).sum(-1))

            z = self.dropout(alpha) @ values # => batchSize × multiNum × r × dk
            z = z.transpose(1,2).reshape(B,self.r,-1) # => batchSize × r × multiNum*dk

            z = self.WO(z) # => batchSize × r × feaSize

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean((((pAlpha_.transpose(-1,-2) @ pAlpha_)*self.ones_minus_I[0])**2))

            z = self.dropout(pAlpha_) @ z # => batchSize × seqLen × feaSize
        elif self.linearLevel=='l2':
            # => batchSize × multiNum × seqLen × r
            pScore = self.pseudoAttn(vx).reshape(B,L,self.multiNum,self.r).transpose(1,2)
            if maskPAD is not None:
                pScore = pScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=1).unsqueeze(dim=-1), -2**15+1)

            pAlpha = F.softmax(pScore, dim=-2).transpose(-1,-2) # => batchSize × multiNum × r × seqLen
            pAlpha_ = F.softmax(pScore, dim=-1) # => batchSize × multiNum × seqLen × r

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean(((pAlpha@pAlpha.transpose(-1,-2))*self.ones_minus_I)**2)

            qx = self.dropout(pAlpha) @ qx.unsqueeze(1) # => batchSize × multiNum × r × feaSize
            kx = self.dropout(pAlpha) @ kx.unsqueeze(1) # => batchSize × multiNum × r × feaSize
            vx = self.dropout(pAlpha) @ vx.unsqueeze(1) # => batchSize × multiNum × r × feaSize

            queries = qx @ self.WQ.weight.reshape(1,self.multiNum,self.dk,C).transpose(-1,-2) + self.WQ.bias.reshape(1,self.multiNum,1,self.dk)
            keys    = kx @ self.WK.weight.reshape(1,self.multiNum,self.dk,C).transpose(-1,-2) + self.WK.bias.reshape(1,self.multiNum,1,self.dk)
            values  = vx @ self.WV.weight.reshape(1,self.multiNum,self.dk,C).transpose(-1,-2) + self.WV.bias.reshape(1,self.multiNum,1,self.dk)
        
            scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × r × r

            # residual attention
            if preScores is not None:
                scores = scores + preScores

            alpha = F.softmax(scores, dim=3) # batchSize × multiNum × r × r

            if qx.requires_grad:
                # add the diagonal loss
                addLoss += torch.mean((alpha * self.ones_minus_I))
            
            z = self.dropout(alpha) @ values # => batchSize × multiNum × r × dk
            
            z = self.dropout(pAlpha_) @ z # => batchSize × multiNum × seqLen × dk

            z = z.transpose(1,2).reshape(B,L,-1) # => batchSize × seqLen × multiNum*dk

            z = self.WO(z) # => batchSize × seqLen × feaSize

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean(((pAlpha_.transpose(-1,-2) @ pAlpha_)*self.ones_minus_I)**2)
        elif self.linearLevel=='l3':
            qkvScore,uScore = self.fQKV(vx),self.fU(vx) # => batchSize × seqLen × r
            if maskPAD is not None:
                qkvScore = qkvScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)
                uScore = uScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)

            qkvAlpha = F.softmax(qkvScore, dim=-2).transpose(-1,-2) # => batchSize × r × seqLen
            uAlpha_ = F.softmax(uScore, dim=-1) # => batchSize × seqLen × r

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean(((qkvAlpha@qkvAlpha.transpose(-1,-2))*self.ones_minus_I[0])**2)

            qx = self.dropout(qkvAlpha) @ qx # => batchSize × r × feaSize
            kx = self.dropout(qkvAlpha) @ kx # => batchSize × r × feaSize
            vx = self.dropout(qkvAlpha) @ vx # => batchSize × r × feaSize

            queries = self.WQ(qx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            keys    = self.WK(kx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            values  = self.WV(vx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
        
            scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × r × r

            # residual attention
            if preScores is not None:
                scores = scores + preScores

            alpha = F.softmax(scores, dim=3) # batchSize × multiNum × r × r

            if qx.requires_grad:
                # add the diagonal loss
                addLoss += torch.mean((alpha * self.ones_minus_I))
            
            z = self.dropout(alpha) @ values # => batchSize × multiNum × r × dk
            z = z.transpose(1,2).reshape(B,self.r,-1) # => batchSize × r × multiNum*dk

            z = self.WO(z) # => batchSize × r × feaSize

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean(((uAlpha_.transpose(-1,-2) @ uAlpha_)*self.ones_minus_I[0])**2)

            z = self.dropout(uAlpha_) @ z # => batchSize × seqLen × feaSize
        elif self.linearLevel=='l4':
            qScore,kScore,uScore,vScore = self.fQ(vx),self.fK(vx),self.fU(vx),self.fV(vx) # => batchSize × seqLen × r
            if maskPAD is not None:
                qScore = qScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)
                kScore = kScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)
                uScore = uScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)
                vScore = vScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)

            qAlpha = F.softmax(qScore, dim=-2).transpose(-1,-2) # => batchSize × r × seqLen
            kAlpha = F.softmax(kScore, dim=-2).transpose(-1,-2) # => batchSize × r × seqLen
            vAlpha = F.softmax(vScore, dim=-2).transpose(-1,-2) # => batchSize × r × seqLen
            uAlpha_ = F.softmax(uScore, dim=-1) # => batchSize × seqLen × r

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean(((vAlpha@vAlpha.transpose(-1,-2))*self.ones_minus_I[0])**2)

            qx = self.dropout(qAlpha) @ qx # => batchSize × r × feaSize
            kx = self.dropout(kAlpha) @ kx # => batchSize × r × feaSize
            vx = self.dropout(vAlpha) @ vx # => batchSize × r × feaSize

            queries = self.WQ(qx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            keys    = self.WK(kx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            values  = self.WV(vx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
        
            scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × r × r

            # residual attention
            if preScores is not None:
                scores = scores + preScores

            alpha = F.softmax(scores, dim=3) # batchSize × multiNum × r × r

            if qx.requires_grad:
                # add the diagonal loss
                addLoss += torch.mean((alpha * self.ones_minus_I))
            
            z = self.dropout(alpha) @ values # => batchSize × multiNum × r × dk
            z = z.transpose(1,2).reshape(B,self.r,-1) # => batchSize × r × multiNum*dk

            z = self.WO(z) # => batchSize × r × feaSize

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean(((uAlpha_.transpose(-1,-2) @ uAlpha_)*self.ones_minus_I[0])**2)

            z = self.dropout(uAlpha_) @ z # => batchSize × seqLen × feaSize
        elif self.linearLevel=='l5':
            # => batchSize × multiNum × seqLen × r
            qScore,kScore,uScore,vScore = self.fQ(vx).reshape(B,L,self.multiNum,self.r).transpose(1,2),self.fK(vx).reshape(B,L,self.multiNum,self.r).transpose(1,2),self.fU(vx).reshape(B,L,self.multiNum,self.r).transpose(1,2),self.fV(vx).reshape(B,L,self.multiNum,self.r).transpose(1,2)
            if maskPAD is not None:
                qScore = qScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=1).unsqueeze(dim=-1), -2**15+1)
                kScore = kScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=1).unsqueeze(dim=-1), -2**15+1)
                uScore = uScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=1).unsqueeze(dim=-1), -2**15+1)
                vScore = vScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=1).unsqueeze(dim=-1), -2**15+1)

            qAlpha = F.softmax(qScore, dim=-2).transpose(-1,-2) # => batchSize × multiNum × r × seqLen
            kAlpha = F.softmax(kScore, dim=-2).transpose(-1,-2) # => batchSize × multiNum × r × seqLen
            vAlpha = F.softmax(vScore, dim=-2).transpose(-1,-2) # => batchSize × multiNum × r × seqLen
            uAlpha_ = F.softmax(uScore, dim=-1) # => batchSize × multiNum × seqLen × r

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean(((vAlpha@vAlpha.transpose(-1,-2))*self.ones_minus_I)**2)

            qx = self.dropout(qAlpha) @ qx.unsqueeze(1) # => batchSize × multiNum × r × feaSize
            kx = self.dropout(kAlpha) @ kx.unsqueeze(1) # => batchSize × multiNum × r × feaSize
            vx = self.dropout(vAlpha) @ vx.unsqueeze(1) # => batchSize × multiNum × r × feaSize

            queries = qx @ self.WQ.weight.reshape(1,self.multiNum,self.dk,C).transpose(-1,-2) + self.WQ.bias.reshape(1,self.multiNum,1,self.dk)
            keys    = kx @ self.WK.weight.reshape(1,self.multiNum,self.dk,C).transpose(-1,-2) + self.WK.bias.reshape(1,self.multiNum,1,self.dk)
            values  = vx @ self.WV.weight.reshape(1,self.multiNum,self.dk,C).transpose(-1,-2) + self.WV.bias.reshape(1,self.multiNum,1,self.dk)
        
            scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × r × r

            # residual attention
            if preScores is not None:
                scores = scores + preScores

            alpha = F.softmax(scores, dim=3) # batchSize × multiNum × r × r

            if qx.requires_grad:
                # add the diagonal loss
                addLoss += torch.mean((alpha * self.ones_minus_I))
            
            z = self.dropout(alpha) @ values # => batchSize × multiNum × r × dk
            
            z = self.dropout(uAlpha_) @ z # => batchSize × multiNum × seqLen × dk

            z = z.transpose(1,2).reshape(B,L,-1) # => batchSize × seqLen × multiNum*dk

            z = self.WO(z) # => batchSize × seqLen × feaSize

            if qx.requires_grad:
                # add the orthogonal loss
                addLoss += torch.mean(((uAlpha_.transpose(-1,-2) @ uAlpha_)*self.ones_minus_I)**2)

        elif self.linearLevel=='l3':
            kvScore = self.fKV(vx) # => batchSize × qL × r, batchSize × kvL × r
            if maskPAD is not None:
                kvScore = kvScore.masked_fill((maskPAD[:,0]==0).unsqueeze(dim=-1), -2**15+1)
                
            kvAlpha = F.softmax(kvScore, dim=-2).transpose(-1,-2) # => batchSize × r × kvL
            kvAlpha_ = F.softmax(kvScore, dim=-1) # => batchSize × kvL × r

            qx = qx # => batchSize × qL × feaSize
            kx = self.dropout(kvAlpha) @ kx # => batchSize × r × feaSize
            vx = self.dropout(kvAlpha) @ vx # => batchSize × r × feaSize

            queries = self.WQ(qx).reshape(B,qL,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × qL × dk
            keys    = self.WK(kx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
            values  = self.WV(vx).reshape(B,self.r,self.multiNum,self.dk).transpose(1,2) # => batchSize × multiNum × r × dk
        
            scores = queries@keys.transpose(-1,-2) / np.sqrt(self.dk) # => batchSize × multiNum × qL × r

            # residual attention
            if preScores is not None:
                scores = scores + preScores

            alpha = F.softmax(scores, dim=3) # batchSize × multiNum × qL × r
            
            z = self.dropout(alpha) @ values # => batchSize × multiNum × qL × dk
            z = z.transpose(1,2).reshape(B,qL,-1) # => batchSize × qL × multiNum*dk

            z = self.WO(z) # => batchSize × qL × feaSize

        return z,scores,addLoss

class SingularformerBlock(nn.Module):
    def __init__(self, r, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.GELU, linearLevel='l1'):
        super(SingularformerBlock, self).__init__()
        self.selfAttn = SingularAttention(r, feaSize, dk, multiNum, dropout, linearLevel=linearLevel)
        self.ffn = FFN_PostLN(feaSize, dropout)
        self._reset_parameters()

    def forward(self, input):
        qx,kx,vx,preScores,maskPAD,addLoss = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        z,preScores,addLoss_ = self.selfAttn(qx,kx,vx,addLoss,preScores,maskPAD) # => batchSize × seqLen × feaSize
        addLoss += addLoss_

        x = self.ffn(vx, z)
        return (x, x, x, preScores,maskPAD,addLoss) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)
class SingularformerLayers(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, r=64, hdnDropout=0.1,  
                 actFunc=nn.GELU, linearLevel='l1', name='textTransformer'):
        super(SingularformerLayers, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, SingularformerBlock(r, feaSize, dk, multiNum, hdnDropout, actFunc, linearLevel)) for i in range(layersNum)]
                                     )
                                 )
        self.layersNum = layersNum
        self.name = name
    def forward(self, x, maskPAD):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD,addLoss = self.transformerLayers((x, x, x, None, maskPAD,0))
        return (qx,kx,vx,scores,maskPAD,addLoss/self.layersNum)# => batchSize × seqLen × feaSize

class MLP(nn.Module):
    def __init__(self, inSize, outSize, hiddenList=[], dropout=0.0, inBn=False, bnEveryLayer=False, outBn=False, 
                                                                    inDp=False, dpEveryLayer=False, outDp=False, name='MLP', 
                                                                    outAct=False, actFunc=nn.ReLU):
        super(MLP, self).__init__()
        self.name = name
        self.sBn = nn.BatchNorm1d(inSize)
        hiddens,bns = [],[]
        for i,os in enumerate(hiddenList):
            hiddens.append( nn.Sequential(
                nn.Linear(inSize, os),
            ) )
            bns.append(nn.BatchNorm1d(os))
            inSize = os
        bns.append(nn.BatchNorm1d(outSize))
        self.actFunc = actFunc()
        self.dropout = nn.Dropout(p=dropout)
        self.hiddens = nn.ModuleList(hiddens)
        self.bns = nn.ModuleList(bns)
        self.out = nn.Linear(inSize, outSize)
        self.bnEveryLayer = bnEveryLayer
        self.dpEveryLayer = dpEveryLayer
        self.inBn = inBn
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.inDp = inDp
    def forward(self, x):
        preShape = x.shape
        if self.inBn:
            x = self.sBn(x)
        if self.inDp:
            x = self.dropout(x)
        for h,bn in zip(self.hiddens,self.bns):
            x = h(x)
            if self.bnEveryLayer:
                x = bn(x) if len(x.shape)==2 else bn(x.transpose(1,2)).transpose(1,2)
            x = self.actFunc(x)
            if self.dpEveryLayer:
                x = self.dropout(x)
        x = self.out(x)
        if self.outBn: x = self.bns[-1](x) if len(x.shape)==2 else self.bns[-1](x.transpose(1,2)).transpose(1,2)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x

class LayerNormAndDropout(nn.Module):
    def __init__(self, feaSize, dropout=0.1, name='layerNormAndDropout'):
        super(LayerNormAndDropout, self).__init__()
        self.layerNorm = nn.LayerNorm([feaSize])
        self.dropout = nn.Dropout(p=dropout)
        self.name = name
    def forward(self, x):
        return self.dropout(self.layerNorm(x))

# for generation model
class SingularformerDecoderBlock(nn.Module):
    def __init__(self, r, feaSize, dk, multiNum, dropout=0.1, actFunc=nn.GELU):
        super(SingularformerDecoderBlock, self).__init__()
        self.selfAttn1 = SelfAttention_PostLN(feaSize, dk, multiNum, dropout)
        self.selfAttn2 = SingularAttention(r, feaSize, dk, multiNum, dropout, linearLevel='l0')
        self.layernorm = nn.LayerNorm([feaSize])
        self.ffn = FFN_Pseudoformer(feaSize, dropout, actFunc)
        self._reset_parameters()

    def forward(self, input, predict=False):
        qx,kx,vx,preScores,maskPAD1,maskPAD2,addLoss = input
        # x: batchSize × seqLen × feaSize; xlen: batchSize
        # zq,preScores,addLoss = self.selfAttn1(qx,qx,qx,addLoss,preScores,maskPAD1) # => batchSize × seqLen × feaSize
        qx = self.layernorm(qx+self.selfAttn1(qx,qx,qx,maskPAD1)) # => batchSize × seqLen × feaSize
        
        zq,preScores,addLoss = self.selfAttn2(qx,kx,vx,addLoss,preScores,maskPAD2) # => batchSize × seqLen × feaSize
        # print(zq.shape,zq[0,[0]])
        # zq = self.selfAttn2(zq,kx,vx,maskPAD2)

        x = self.ffn(qx, zq)
        return (x, kx, vx, preScores,maskPAD1,maskPAD2,addLoss) # => batchSize × seqLen × feaSize

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name,p in self.named_parameters():
            if 'weight' in name and len(p.shape) > 1:
                p.data = truncated_normal_(p.data, std=0.02)
            elif 'bias' in name:
                p.data.fill_(0)

class SingularformerDecoderLayers(nn.Module):
    def __init__(self, layersNum, feaSize, dk, multiNum, r=64, hdnDropout=0.1,  
                 actFunc=nn.GELU, linearLevel='l1', name='textTransformer'):
        super(SingularformerDecoderLayers, self).__init__()
        self.transformerLayers = nn.Sequential(
                                     OrderedDict(
                                         [('transformer%d'%i, SingularformerDecoderBlock(r, feaSize, dk, multiNum, hdnDropout, actFunc, linearLevel)) for i in range(layersNum)]
                                     )
                                 )
        self.layersNum = layersNum
        self.name = name
    def forward(self, x, xRef, maskPAD1, maskPAD2):
        # x: batchSize × seqLen × feaSize; 
        qx,kx,vx,scores,maskPAD1,maskPAD2,addLoss = self.transformerLayers((x, xRef, xRef, None, maskPAD1,maskPAD2,0))
        return (qx,kx,vx,scores,maskPAD1,maskPAD2,addLoss/self.layersNum)# => batchSize × seqLen × feaSize