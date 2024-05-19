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

qm7 = scipy.io.loadmat('../Data/qm7.mat')
X,T,P,Z,R = qm7['X'], qm7['T'], qm7['P'], qm7['Z'], qm7['R'] 
y = np.transpose(qm7['T']).reshape((7165,))
y = y/2000 



index = 0
test_X = X[P[index]]
test_y = y[P[index]]
P_train = np.stack(tuple(P[i] for i in range(5) if i != index), axis = 0)


class QM7_data:
    def __init__(self, X = X, y = y, scale_data=True, mode = 'train'):
        if mode == "train":
            P_train = np.stack(tuple(P[i] for i in range(5) if i != index), axis = 0)
            self.X = np.concatenate(tuple(X[train] for train in P_train), axis = 0)
            self.y = np.concatenate(tuple(y[train] for train in P_train), axis = 0)
        if mode == 'test':
            self.X = test_X
            self.y = test_y 
    
    def __len__(self):
        return len(self.X)
    def augumented(self, data):
        ori = data 
        ori_1 = np.rot90(ori, -1)
        ori_2 = np.rot90(ori_1, -1)
        ori_3 = np.rot90(ori_2, -1)
        return np.stack((ori, ori_1, ori_2, ori_3), axis = 0)
    
    def __getitem__(self, idx):
        return torch.tensor(self.augumented(self.X[idx]), device=device, dtype = torch.float32), torch.tensor(self.y[idx], device=device)
    


dataset = QM7_data()

trainloader = DataLoader(dataset, batch_size=256, shuffle=False)
testloader = DataLoader(QM7_data(mode = 'test'), batch_size=1433, shuffle=False)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential (
        nn.Conv2d(4, 8, 3, padding = 3//2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(8),
        nn.Conv2d(8, 16, 3, padding = 3//2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(16),
        nn.Flatten()
        )
        self.MLP = nn.Sequential (
        nn.Linear(16*5*5, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, 1)
        )

    def forward(self, x):
        output = self.layers(x)
        output = self.MLP(output)
        return output 
    


import torch.nn.functional as F
loss_function = F.mse_loss

cnn = Model().to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-2)
for epoch in range(1000):
    current_loss = 0.0

    for i, data in enumerate(trainloader):
        inputs, targets = data
        targets = targets.view(-1,1)

        outputs = cnn(inputs)
        loss = loss_function(outputs, targets)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        current_loss += loss.item()

    print(f'Epoch {epoch+1} - Loss {current_loss/(i+1)}')
    torch.save(cnn.state_dict(), os.path.join('cnn.pt'))
    for data in testloader:
        inputs, targets = data
        targets = targets.view(-1,1)
        outputs = cnn(inputs)
    print(loss_function(outputs, targets)**0.5)


print("Training has completed")
    



