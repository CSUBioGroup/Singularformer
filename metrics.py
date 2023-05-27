import numpy as np
from sklearn import metrics as skmetrics
import warnings
warnings.filterwarnings("ignore")

def lgb_MaF(preds, dtrain):
    Y = np.array(dtrain.get_label(), dtype=np.int32)
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'macro_f1', float(F1(preds.shape[0], Y_pre, Y, 'macro')), True

def lgb_precision(preds, dtrain):
    Y = dtrain.get_label()
    preds = preds.reshape(-1,len(Y))
    Y_pre = np.argmax( preds, axis=0 )
    return 'precision', float(Counter(Y==Y_pre)[True]/len(Y)), True
    
class Metrictor:
    def __init__(self, multiLabel=False):
        self._reporter_ = {"ACC":self.ACC, "MiF":self.MiF, "MaF":self.MaF}
        self.multiLabel = multiLabel
    def __call__(self, report, end='\n'):
        res = {}
        for mtc in report:
            v = self._reporter_[mtc]()
            print(f" {mtc}={v:6.3f}", end=';')
            res[mtc] = v
        print(end=end)
        return res
    
    def set_data(self, res):
        self.Y = res['y_true']
        self.Y_prob = res['y_prob']
        if self.multiLabel:
            self.Y_pre = (res['y_prob']>0.5).astype('int32')
        else:
            self.Y_pre = np.argmax(res['y_prob'], axis=-1)

    @staticmethod
    def table_show(resList, report, rowName='CV'):
        lineLen = len(report)*8 + 6
        print("="*(lineLen//2-6) + "FINAL RESULT" + "="*(lineLen//2-6))
        print(f"{'-':^6}" + "".join([f"{i:>8}" for i in report]))
        for i,res in enumerate(resList):
            print(f"{rowName+'_'+str(i+1):^6}" + "".join([f"{res[j]:>8.3f}" for j in report]))
        print(f"{'MEAN':^6}" + "".join([f"{np.mean([res[i] for res in resList]):>8.3f}" for i in report]))
        print("======" + "========"*len(report))
    def each_class_indictor_show(self, id2lab):
        print('Waiting for finishing...')

    def ACC(self):
        if self.multiLabel:
            isValid = self.Y.sum(axis=1)>0
            I = (self.Y&self.Y_pre).sum(axis=-1)
            U = (self.Y|self.Y_pre).sum(axis=-1)
            return (np.sum(I[isValid]/U[isValid]) + np.sum(I[~isValid]==0)) / len(I)
        else:
            return np.sum(self.Y==self.Y_pre) / len(self.Y)
    def MiF(self):
        return skmetrics.f1_score(self.Y, self.Y_pre, average='micro')
    def MaF(self):
        return skmetrics.f1_score(self.Y, self.Y_pre, average='macro')