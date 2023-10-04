import torch
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU,ELU
from torch.nn import LogSoftmax
from torch import flatten
from torch.nn import Sequential
from torch.nn.functional import normalize

class Encoder(Module):
    def __init__(self, channels, outDims):

        super(Encoder, self).__init__()
        self.feature_extraction = Sequential(
            Conv2d(in_channels = channels, out_channels = 32, kernel_size = (3,3)),
            ReLU()
            # MaxPool2d(kernel_size = (2,2))
        )
        self.fc = Linear(in_features = 32*30*30, out_features = outDims )
    def forward(self,x):
        x = self.feature_extraction(x)
        # import pdb;pdb.set_trace()
        x = x.reshape((x.size(0),32*30*30))
        x = self.fc(x)
        x = normalize(x)
        return x
    
class Mlp(Module):
    def __init__(self, inDims, outDims):

        super(Mlp,self).__init__()
        self.mlp = Sequential(
            Linear(in_features=inDims, out_features=outDims),
            ELU()
        )

    def forward(self,x):
        x=self.mlp(x)
        x=normalize(x)
        return x

    

if __name__ == '__main__':
    model = Encoder(channels=3,outDims=64)
    print (model)
    input = torch.randn(10,3,32,32)
    out = model(input)
    import pdb;pdb.set_trace()
        