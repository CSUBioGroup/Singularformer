from DL_ClassifierModel import *
from utils import *
from collateFuncs import *
from tokenizerFuncs import *
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit,StratifiedKFold, KFold
import pandas as pd
import argparse
import pynvml

def get_gpu_used_mem():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.used/1024**2  #已用显存大小

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_backbone(modelName, tokenizer):
    if modelName.startswith('CNN'):
        h = int(modelName[3:])
        backbone = BaselineCNN(classNum=len(tokenizer.id2lab),
                               embedding=tokenizer.embedding,
                               hiddenSize=h, contextSizeList=[1,5,25], 
                               embDropout=0.2, hdnDropout=0.2).cuda()
    elif modelName.startswith('RNN'):
        h = int(modelName[3:])
        backbone = BaselineRNN(classNum=len(tokenizer.id2lab),
                               embedding=tokenizer.embedding,
                               hiddenSize=h,
                               embDropout=0.2, hdnDropout=0.2).cuda() 
    elif modelName.startswith('Transformer_PostLN'):
        L,H,A = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)$', modelName)[0]
        backbone = BaselineTransformer(classNum=len(tokenizer.id2lab), LN='post', seqMaxLen=tokenizer.seqMaxLen,
                                       embedding=tokenizer.embedding,
                                       L=int(L), H=int(H), A=int(A), 
                                       embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('Transformer_PreLN'):
        L,H,A = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)$', modelName)[0]
        backbone = BaselineTransformer(classNum=len(tokenizer.id2lab), LN='pre', seqMaxLen=tokenizer.seqMaxLen,
                                       embedding=tokenizer.embedding,
                                       L=int(L), H=int(H), A=int(A), 
                                       embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('Realformer'):
        L,H,A = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)$', modelName)[0]
        backbone = BaselineTransformer(classNum=len(tokenizer.id2lab), LN='realformer', seqMaxLen=tokenizer.seqMaxLen,
                                       embedding=tokenizer.embedding,
                                       L=int(L), H=int(H), A=int(A), 
                                       embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('Linformer_'):
        L,H,A,p = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)_p(.*?)$', modelName)[0]
        backbone = Linformer(classNum=len(tokenizer.id2lab), seqMaxLen=tokenizer.seqMaxLen, 
                             embedding=tokenizer.embedding, 
                             pseTknNum=int(p),
                             L=int(L), H=int(H), A=int(A), 
                             embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('Linformer2_'):
        L,H,A,p = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)_p(.*?)$', modelName)[0]
        backbone = Linformer2(classNum=len(tokenizer.id2lab), seqMaxLen=tokenizer.seqMaxLen, 
                              embedding=tokenizer.embedding, 
                              pseTknNum=int(p),
                              L=int(L), H=int(H), A=int(A), 
                              embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('Singularformer'):
        L,H,A,r,l,g = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)_r([0-9]*?)_l([0-9])_g(.*?)$', modelName)[0]
        backbone = Singularformer(classNum=len(tokenizer.id2lab), seqMaxLen=tokenizer.seqMaxLen, 
                                  embedding=tokenizer.embedding, 
                                  r=int(r), gama=float(g), linearLevel=f'l{l}',
                                  L=int(L), H=int(H), A=int(A), 
                                  embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('Pseformer2_'):
        L,H,A,p,g = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)_p([0-9]*?)_g(.*?)$', modelName)[0]
        backbone = Pseformer2(classNum=len(tokenizer.id2lab), seqMaxLen=tokenizer.seqMaxLen, 
                              embedding=tokenizer.embedding, 
                              pseTknNum=int(p), gama=float(g),
                              L=int(L), H=int(H), A=int(A), 
                              embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('Pseformer_'):
        L,H,A,p,g = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)_p([0-9]*?)_g(.*?)$', modelName)[0]
        backbone = Pseformer(classNum=len(tokenizer.id2lab), seqMaxLen=tokenizer.seqMaxLen, 
                             embedding=tokenizer.embedding, 
                             pseTknNum=int(p), gama=float(g),
                             L=int(L), H=int(H), A=int(A), 
                             embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('BigBird_'):
        L,H,A,r = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)_r(.*?)$', modelName)[0]
        backbone = BigBird(classNum=len(tokenizer.id2lab), 
                           embedding=tokenizer.embedding, 
                           randomK=int(r),
                           L=int(L), H=int(H), A=int(A), 
                           embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('BigBird2_'):
        L,H,A = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)$', modelName)[0]
        backbone = BigBird2(classNum=len(tokenizer.id2lab), seqMaxLen=tokenizer.seqMaxLen,
                            embedding=tokenizer.embedding,
                            L=int(L), H=int(H), A=int(A), 
                            embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('LSTransformer_'):
        L,H,A,r,w = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)_r(.*?)_w(.*?)$', modelName)[0]
        backbone = LSTransformer(classNum=len(tokenizer.id2lab), seqMaxLen=tokenizer.seqMaxLen, 
                                 embedding=tokenizer.embedding, 
                                 r=int(r), w=int(w),
                                 L=int(L), H=int(H), A=int(A), 
                                 embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('FLASH_'):
        L,H,A = re.findall('L([0-9]*?)H([0-9]*?)A([0-9]*?)$', modelName)[0]
        backbone = FLASH(classNum=len(tokenizer.id2lab), seqMaxLen=tokenizer.seqMaxLen, 
                         embedding=tokenizer.embedding, 
                         L=int(L), H=int(H), A=int(A), 
                         embDropout=0.2, hdnDropout=0.1).cuda()
    elif modelName.startswith('ResNet18'):
        backbone = ResNet18(classNum=len(tokenizer.id2lab), pretrained=True, wh=32).cuda()

    return backbone

def get_dataset(datasetName, seqMaxLen=None, isBigBird=False):
    if datasetName.startswith('20NG'):
        if seqMaxLen is None:
            seqMaxLen = 700
        if isBigBird: seqMaxLen = int((1+seqMaxLen//16)*16)
        root = "./datasets/20NG"
        trainDS = Sklearn_20NG(root, 'train')
        testDS = Sklearn_20NG(root, 'test')
        tokenizer = Tokenizer(trainDS.sequences+testDS.sequences, trainDS.labels+testDS.labels, seqMaxLen=seqMaxLen)
        tokenizer.vectorize(trainDS.sequences+testDS.sequences, method=['skipgram','glove','fasttext'], embSize=128, loadCache=True, suf='NG')
    elif datasetName.startswith('BBC_News'):
        if seqMaxLen is None:
            seqMaxLen = 664
        if isBigBird: seqMaxLen = int((1+seqMaxLen//16)*16)
        root = "./datasets/BBC_News"
        allBBC = BBC_News(os.path.join(root, 'bbc-text.csv'))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=9527)
        for trainIdx, testIdx in sss.split(range(len(allBBC)), allBBC.labels):
            break
        trainDS = allBBC.get_sub_set(trainIdx)
        testDS = allBBC.get_sub_set(testIdx)
        tokenizer = Tokenizer(allBBC.sequences, allBBC.labels, seqMaxLen=seqMaxLen)
        tokenizer.vectorize(allBBC.sequences, method=['skipgram','glove','fasttext'], embSize=128, loadCache=True, suf='BBC')
    elif datasetName.startswith('ASTRAL_SCOPe40'):
        if seqMaxLen is None:
            seqMaxLen = 337
        if isBigBird: seqMaxLen = int((1+seqMaxLen//16)*16)
        root = "./datasets/ASTRAL_SCOPe"
        allSCOPe = ASTRAL_SCOPe(os.path.join(root, 'astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa.txt'))

        isValidLab = {k:v>1 for k,v in Counter(allSCOPe.labels).items()}
        validIdx = np.arange(len(allSCOPe))[list(map(lambda x:isValidLab[x], allSCOPe.labels))]
        allSCOPe = allSCOPe.get_sub_set(validIdx)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=9527)
        for trainIdx, testIdx in sss.split(range(len(allSCOPe)), allSCOPe.labels):
            break
        trainDS = allSCOPe.get_sub_set(trainIdx)
        testDS = allSCOPe.get_sub_set(testIdx)

        tokenizer = Tokenizer(allSCOPe.sequences, allSCOPe.labels, seqMaxLen=seqMaxLen)
        tokenizer.vectorize(allSCOPe.sequences, method=['skipgram','glove','fasttext'], embSize=128, loadCache=True, suf='ASTRAL_SCOPe40')
    elif datasetName.startswith('ASTRAL_SCOPe95'):
        if seqMaxLen is None:
            seqMaxLen = 328
        if isBigBird: seqMaxLen = int((1+seqMaxLen//16)*16)
        root = "./datasets/ASTRAL_SCOPe"
        allSCOPe = ASTRAL_SCOPe(os.path.join(root, 'astral-scopedom-seqres-gd-sel-gs-bib-95-2.08.fa.txt'))

        isValidLab = {k:v>1 for k,v in Counter(allSCOPe.labels).items()}
        validIdx = np.arange(len(allSCOPe))[list(map(lambda x:isValidLab[x], allSCOPe.labels))]
        allSCOPe = allSCOPe.get_sub_set(validIdx)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=9527)
        for trainIdx, testIdx in sss.split(range(len(allSCOPe)), allSCOPe.labels):
            break
        trainDS = allSCOPe.get_sub_set(trainIdx)
        testDS = allSCOPe.get_sub_set(testIdx)

        tokenizer = Tokenizer(allSCOPe.sequences, allSCOPe.labels, seqMaxLen=seqMaxLen)
        tokenizer.vectorize(allSCOPe.sequences, method=['skipgram','glove','fasttext'], embSize=128, loadCache=True, suf='ASTRAL_SCOPe95')
    elif datasetName.startswith('IMDB'):
        if seqMaxLen is None:
            seqMaxLen = 539
        if isBigBird: seqMaxLen = int((1+seqMaxLen//16)*16)
        root = "./datasets/IMDB/aclImdb"
        trainDS = IMDB(os.path.join(root, 'train'))
        testDS = IMDB(os.path.join(root, 'test'))
        tokenizer = Tokenizer(trainDS.sequences+testDS.sequences, trainDS.labels+testDS.labels, seqMaxLen=seqMaxLen)
        tokenizer.vectorize(trainDS.sequences+testDS.sequences, method=['skipgram','glove','fasttext'], embSize=128, loadCache=True, suf='IMDB')
    elif datasetName.startswith('R8'):
        if seqMaxLen is None:
            seqMaxLen = 141
        if isBigBird: seqMaxLen = int((1+seqMaxLen//16)*16)
        root = "./datasets/R8R52OH/r8"
        trainDS = R8(os.path.join(root, 'r8-train-stemmed.csv'))
        validDS = R8(os.path.join(root, 'r8-dev-stemmed.csv'))
        testDS = R8(os.path.join(root, 'r8-test-stemmed.csv'))
        trainDS = trainDS+validDS
        tokenizer = Tokenizer(trainDS.sequences+testDS.sequences, trainDS.labels+testDS.labels, seqMaxLen=seqMaxLen)
        tokenizer.vectorize(trainDS.sequences+testDS.sequences, method=['skipgram','glove','fasttext'], embSize=128, loadCache=True, suf='R8')
    elif datasetName.startswith('MIMIC3'):
        if seqMaxLen is None:
            seqMaxLen = 1024
        if isBigBird: seqMaxLen = int((1+seqMaxLen//16)*16)
        root = "./datasets/MIMIC3/"
        allMIMIC3 = MIMIC3(os.path.join(root, 'all.csv'))
        ctr = Counter()
        for i in tqdm(allMIMIC3.labels):
            ctr += Counter(i)
        selectedLabels = set([i[0] for i in ctr.most_common(50)])
        trainDS = MIMIC3(os.path.join(root,'train.csv'), selectedLabels)
        validDS = MIMIC3(os.path.join(root,'valid.csv'), selectedLabels)
        testDS = MIMIC3(os.path.join(root,'test.csv'), selectedLabels)
        tokenizer = Tokenizer(trainDS.sequences+validDS.sequences+testDS.sequences, trainDS.labels+validDS.labels+testDS.labels, seqMaxLen=seqMaxLen, multiLabel=True) # 2978
        tokenizer.vectorize(trainDS.sequences+validDS.sequences+testDS.sequences, method=['skipgram','glove','fasttext'], embSize=128, loadCache=True, suf='MIMIC3_50')
        trainDS.cache_tokenized_input(tokenizer)
        validDS.cache_tokenized_input(tokenizer)
        testDS.cache_tokenized_input(tokenizer)
        trainDS = trainDS+validDS
    elif datasetName.startswith('ListOps'):
        if seqMaxLen is None:
            seqMaxLen = 4912
        if isBigBird: seqMaxLen = int((1+seqMaxLen//16)*16)
        root = "./datasets/lra_data/listops"
        trainDS = ListOps(os.path.join(root, 'basic_train.tsv'))
        validDS = ListOps(os.path.join(root, 'basic_val.tsv'))
        testDS = ListOps(os.path.join(root, 'basic_test.tsv'))
        trainDS = trainDS+validDS
        tokenizer = Tokenizer(trainDS.sequences+testDS.sequences, trainDS.labels+testDS.labels, seqMaxLen=seqMaxLen)
        tokenizer.vectorize(trainDS.sequences+testDS.sequences, method=['skipgram','glove','fasttext'], embSize=128, loadCache=True, suf='ListOps')
    elif datasetName.startswith('CIFAR_100'):
        seqMaxLen = 256
        root = "./datasets/CIFAR_100"
        trainDS = CIFAR_100(os.path.join(root,'train'), seqMaxLen=seqMaxLen)
        testDS = CIFAR_100(os.path.join(root,'test'), seqMaxLen=seqMaxLen)
        tokenizer = Tokenizer(['xxx'], trainDS.id2lab, seqMaxLen=seqMaxLen)
        tokenizer.seqMaxLen = seqMaxLen
        # tokenizer.embedding = VisioEmbedding(embDropout=0.2) # nn.Linear(3,384)
        tokenizer.embedding = InputCNN(embDropout=0.2)

    return trainDS, testDS, tokenizer

# CUDA_VISIBLE_DEVICES=1 python train_main.py --models "CNN64;RNN64;TransformerL4H256A4;PseformerL4H256A4_p16;PseformerL4H256A4_p64" --datasets "20NG;BBC_News;ASTRAL_SCOPe40;ASTRAL_SCOPe95;IMDB;R8"

parser = argparse.ArgumentParser()
parser.add_argument('--models', default="PseformerL4H256A4_p64", type=str)
parser.add_argument('--datasets', default="R8", type=str)
parser.add_argument('--lr', default=3e-4, type=float)
parser.add_argument('--batchSize', default=64, type=int)
parser.add_argument('--earlyStop', default=32, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--warmup', default=4, type=int)

parser.add_argument('--seqMaxLen', default=-1, type=int)
parser.add_argument('--dataLoadNumWorkers', default=0, type=int)
parser.add_argument('--prefetchFactor', default=2, type=int)

parser.add_argument('--nSplits', default=5, type=int)
parser.add_argument('--cv', default=-1, type=int)
args = parser.parse_args()

if __name__ == '__main__':
    models = args.models.split(';')
    datasets = args.datasets.split(';')
    if args.seqMaxLen<1:
        args.seqMaxLen = None

    for datasetName in datasets:
        totalDS,testDS,tokenizer = get_dataset(datasetName, args.seqMaxLen, np.any(['BigBird2' in mn for mn in models]))

        for modelName in models:

            if datasetName.startswith('MIMIC3'):
                multiLabel = True
                kf = KFold(n_splits=args.nSplits, shuffle=True, random_state=9527)
                kfold = kf.split(range(len(totalDS)))
            else:
                multiLabel = False
                skf = StratifiedKFold(n_splits=args.nSplits, shuffle=True, random_state=9527)
                kfold = skf.split(range(len(totalDS)), totalDS.labels)
            
            for k, (trainIdx,validIdx) in enumerate(kfold):
                if args.cv>0 and (k+1)!=args.cv: continue
                trainDS,validDS = torch.utils.data.Subset(totalDS, trainIdx),torch.utils.data.Subset(totalDS, validIdx)

                torch.cuda.empty_cache()

                print(f'\n\n====================\nTraining model {modelName} in dataset {datasetName}...\n====================\n\n')

                set_seed(9527)
                backbone = get_backbone(modelName, tokenizer)

                CEL = nn.CrossEntropyLoss()

                lr = args.lr
                pretrain = ['embedding']
                if 'ResNet' in modelName:
                    pretrain += ['conv','bn','layer']
                no_decay = ['bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in backbone.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad and any(pt in n for pt in pretrain)],
                     'weight_decay': 0.001, 'lr': lr/3}, # 
                    {'params': [p for n, p in backbone.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad and (not any(pt in n for pt in pretrain))],
                     'weight_decay': 0.001, 'lr': lr},
                    {'params': [p for n, p in backbone.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad  and any(pt in n for pt in pretrain)], 
                     'weight_decay': 0.0, 'lr': lr/3}, #  
                    {'params': [p for n, p in backbone.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad  and (not any(pt in n for pt in pretrain))], 
                     'weight_decay': 0.0, 'lr': lr}, 
                ]
                optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=lr, weight_decay=0.0)

                if multiLabel:
                    model = SequenceMultiLabelClassification(backbone, criterion=None, AMP=True,
                                                             collateFunc=PadAndTknizeCollateFunc(tokenizer, baseLen=backbone.w if 'LSTrans' in modelName else 1))
                    model.metrictor = Metrictor(multiLabel=True)
                else:
                    model = SequenceClassification(backbone, criterion=CEL, AMP=True,
                                                   collateFunc=PadAndTknizeCollateFunc(tokenizer, inputType=torch.float32 if 'CIFAR' in datasetName else torch.long,
                                                                                       baseLen=backbone.w if 'LSTrans' in modelName else 1))

                evalSteps = len(trainDS)//args.batchSize
                model.train(optimizer, trainDataSet=trainDS, validDataSet=validDS, # trainDS
                            batchSize=args.batchSize, maxSteps=evalSteps*args.epochs, evalSteps=evalSteps, earlyStop=args.earlyStop, saveSteps=-1,  # evalSteps*10000
                            isHigherBetter=True, metrics="ACC", report=["ACC"],   
                            savePath=f'trained_models/{modelName}_for_{datasetName}_cv{k+1}', dataLoadNumWorkers=args.dataLoadNumWorkers, pinMemory=True, 
                            warmupSteps=evalSteps*args.warmup, doEvalTrain=False, prefetchFactor=args.prefetchFactor)

# CUDA_VISIBLE_DEVICES=1 python train_main.py --models Pseformer_L6H192A6_p32 --datasets MIMIC3;20NG;BBC_News;ASTRAL_SCOPe40;ASTRAL_SCOPe95;IMDB;R8
# CUDA_VISIBLE_DEVICES=0 python train_main.py --models Linformer_L6H192A6_p32 --datasets MIMIC3;20NG;BBC_News;ASTRAL_SCOPe40;ASTRAL_SCOPe95;IMDB;R8


# CUDA_VISIBLE_DEVICES=0 python -u train_main.py --lr 1e-4 --epochs 256 --earlyStop 256 --dataLoadNumWorkers 4 --prefetchFactor 16 --models Linformer_L6H384A6_p64 --datasets "20NG;BBC_News;ASTRAL_SCOPe40;ASTRAL_SCOPe95;IMDB;R8" > trained_models/Linformer_L6H384A6_p64_loss.log

# python -u train_main.py --lr 1e-4 --epochs 256 --earlyStop 256 --models Pseformer_L6H384A6_p64_g0.1 --datasets "CIFAR_100" --dataLoadNumWorkers 4 --prefetchFactor 16