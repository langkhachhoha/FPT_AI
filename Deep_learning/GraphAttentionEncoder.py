import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import math 
import numpy as np



class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim, # đầu vào
            embed_dim, # đầu ra
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv) # đưa về lấy mâu trên phân phối đều trong khoảng -stdv, stdv

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency) mask = [1,0]
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)


        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=128
    ):
        super(GraphAttentionEncoder, self).__init__()
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        assert mask is None, "TODO mask not yet supported!"

        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return h  # (batch_size, graph_size, embed_dim)
    

# print(model(input).shape)
from torch import nn
import torch 


import scipy.io
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import distance
import torch
from torch import nn
from torch.utils.data import DataLoader
import os 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

qm7 = scipy.io.loadmat('./Data/qm7.mat')
X,T,P,Z,R = qm7['X'], qm7['T'], qm7['P'], qm7['Z'], qm7['R'] 
y = np.transpose(qm7['T']).reshape((7165,))
y = y/2000 



index = 0
test_X = X[P[index]]
test_y = y[P[index]]
test_R = R[P[index]]
P_train = np.stack(tuple(P[i] for i in range(5) if i != index), axis = 0)


class QM7_data:
    def __init__(self, X = X, y = y, R = R, scale_data=True, mode = 'train'):
        if mode == "train":
            P_train = np.stack(tuple(P[i] for i in range(5) if i != index), axis = 0)
            self.X = np.concatenate(tuple(X[train] for train in P_train), axis = 0)
            self.y = np.concatenate(tuple(y[train] for train in P_train), axis = 0)
            self.R = np.concatenate(tuple(R[train] for train in P_train), axis = 0)
        if mode == 'test':
            self.X = test_X
            self.y = test_y 
            self.R = test_R 
    
    def __len__(self):
        return len(self.X)
    def augumented(self, data):
        ori = data 
        ori_1 = np.rot90(ori, -1)
        ori_2 = np.rot90(ori_1, -1)
        ori_3 = np.rot90(ori_2, -1)
        return np.stack((ori, ori_1, ori_2, ori_3), axis = 0)
    
    def __getitem__(self, idx):
        # print(self.R[idx])
        # print(self.X[idx][0][0])
        b = np.array( [self.X[idx][i][i] for i in range(self.X[idx].shape[0])] ).reshape(-1,1)
        c = np.concatenate((self.R[idx], b), axis = 1)
        return torch.tensor(c, device=device, dtype = torch.float32),torch.tensor(self.augumented(self.X[idx]), device=device, dtype = torch.float32), torch.tensor(self.y[idx], device=device)
    


dataset = QM7_data()

trainloader = DataLoader(dataset, batch_size=512, shuffle=True)
testloader = DataLoader(QM7_data(mode = 'test'), batch_size=1433, shuffle=False)




# Model 
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = GraphAttentionEncoder(8, 64, 3, 4)
        self.layers = nn.Sequential (
        nn.Conv2d(4, 16, 5, padding = 5//2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 5, padding = 5//2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, padding = 3//2),
        nn.MaxPool2d(2),
        nn.Flatten()
        )
        self.MLP = nn.Sequential (
        nn.Linear(640, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, 1)
        )

    def forward(self, x, y):

        output = self.encoder(x)
        output = output.unsqueeze(1)
        output = output.expand(-1,4,-1,-1)
        output = torch.concat((y, output), dim = 3)
        output = self.layers(output)
        output = self.MLP(output)

        return output 
    
model = Model().to(device)
import torch.nn.functional as F
loss_function = F.mse_loss


optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(5000):
    current_loss = 0.0

    for i, data in enumerate(trainloader):
        x, y, targets = data
        targets = targets.view(-1,1)

        outputs = model(x,y)
        loss = loss_function(outputs, targets)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        current_loss += loss.item()

    print(f'Epoch {epoch+1} - Loss {current_loss/(i+1)}')
    torch.save(model.state_dict(), os.path.join('gnn.pt'))
    for data in testloader:
        x,y, targets = data
        targets = targets.view(-1,1)
        outputs = model(x,y)
    print(loss_function(outputs, targets)**0.5)

print("Training has completed")
    

     