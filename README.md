## Usage

```python
from nnLayer import *

L,H,A = 4,256,4 # number of layers, number of hidden units, number of attention heads
singularformer = SingularformerLayers(layersNum=L, feaSize=H, dk=H//A, multiNum=A, r=H//A, hdnDropout=0.1)

x = torch.randn((16,1024,256)) # embedded input with the shape (batch size,seq length,embedding size)
output = singularformer(x, None) # output with the shape (batch size,seq length,embedding size)
```
