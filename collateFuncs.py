import torch,random
import numpy as np

class PadAndTknizeCollateFunc:
    def __init__(self, tokenizer, inputType=torch.long, baseLen=1):
        self.tokenizer = tokenizer
        self.eval = False
        self.baseLen = baseLen
        self.inputType = inputType
    def __call__(self, data):
        if 'tokenizedSeqArr' in data[0] and data[0]['tokenizedSeqArr'] is not None:
            tokenizedSeqArr,maskPAD = [i['tokenizedSeqArr'] for i in data],[i['maskPAD'] for i in data]
        else:
            tokenizedSeqArr,maskPAD = self.tokenizer.tokenize_sequences([i['sequence'] for i in data], self.eval, self.baseLen) # batchSize × seqLen
        tokenizedSeqArr,maskPAD = torch.tensor(tokenizedSeqArr, dtype=self.inputType),torch.tensor(maskPAD, dtype=torch.bool)
        # maskPAD = maskPAD.reshape(len(tokenizedSeqArr), 1, -1) & maskPAD.reshape(len(tokenizedSeqArr), -1, 1)

        if 'tokenizedLabArr' in data[0] and data[0]['tokenizedLabArr'] is not None:
            tokenizedLabArr = [i['tokenizedLabArr'] for i in data]
        else:
            tokenizedLabArr = self.tokenizer.tokenize_labels([i['label'] for i in data])
        tokenizedLabArr = torch.tensor(tokenizedLabArr, dtype=torch.long) # batchSize

        return {'tokenizedSeqArr':tokenizedSeqArr, 'maskPAD':maskPAD, 'tokenizedLabArr':tokenizedLabArr}

from transformers import AutoTokenizer, AutoModelForMaskedLM
class PadAndTknizeCollateFunc_huggingface:
    def __init__(self, tokenizer, tknPath='./pretrained_models/roberta-base', maxLength=510):
        self.tokenizer = tokenizer
        self.hgfTkner = AutoTokenizer.from_pretrained(tknPath)
        self.eval = False
        self.maxLength = maxLength
    def __call__(self, data):
        inputData = self.hgfTkner([" ".join(i['sequence']) for i in data], max_length=self.maxLength, 
                                  padding='max_length' if self.eval else 'longest', return_tensors='pt', truncation=True)

        if 'tokenizedLabArr' in data[0] and data[0]['tokenizedLabArr'] is not None:
            tokenizedLabArr = [i['tokenizedLabArr'] for i in data]
        else:
            tokenizedLabArr = self.tokenizer.tokenize_labels([i['label'] for i in data])
        tokenizedLabArr = torch.tensor(tokenizedLabArr, dtype=torch.long) # batchSize

        return {'inputData':inputData, 'tokenizedLabArr':tokenizedLabArr}