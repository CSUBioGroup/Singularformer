# Singularformer: Learning to Decompose Self-Attention to Linearize the Complexity of Transformer

## Overview
Singularformer is a linear-complexity transformer based on singular value decomposition. The idea is from the equation "A=softmax(QK^T)" in Transformer, where "A" is obviously a singular or non-full-rank matrix and the rank is at most H/A (hidden size over number of attention heads). 

## Usage

```python
from nnLayer import *

L,H,A = 4,256,4 # number of layers, number of hidden units, number of attention heads
singularformer = SingularformerLayers(layersNum=L, feaSize=H, dk=H//A, multiNum=A, r=H//A, hdnDropout=0.1)

x = torch.randn((16,1024,256)) # embedded input with the shape (batch size,seq length,embedding size)
output = singularformer(x, None) # output with the shape (batch size,seq length,embedding size)
```
