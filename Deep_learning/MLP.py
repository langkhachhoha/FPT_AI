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
y = y/2000 # atomization energy

# Read Data
num_atoms = X.shape[1]
final_data = []
max_len = num_atoms*num_atoms + 2
for id, molecue in enumerate(X):
    lst = []
    dis = []
    distance_matrix = distance.cdist(R[id], R[id], 'euclidean')
    for i in range(num_atoms):
        if molecue[i][i] == 0:
            break
        lst.append(molecue[i][i])
    lst.append(0)
    for i in range(num_atoms):
        if molecue[i][i] == 0:
            break 
        for j in range(i+1, num_atoms):
            if molecue[i][j] == 0:
                break
            lst.append(molecue[i][j])
            dis.append(distance_matrix[i][j])
    lt = lst+[0]+dis
    while (len(lt) < max_len):
        lt.append(0)
    final_data.append(lt)
transform_data = np.array(final_data)
transform_data = transform_data/100 
input_size = transform_data.shape[1]

index = 0
test_X = transform_data[P[index]]
test_y = y[P[index]]
P_train = np.stack(tuple(P[i] for i in range(5) if i != index), axis = 0)


class QM7_data:
    def __init__(self, X = transform_data, y = y, scale_data=True, mode = 'train'):
        if mode == "train":
            P_train = np.stack(tuple(P[i] for i in range(5) if i != index), axis = 0)
            self.X = np.concatenate(tuple(transform_data[train] for train in P_train), axis = 0)
            self.y = np.concatenate(tuple(y[train] for train in P_train), axis = 0)
        if mode == 'test':
            self.X = test_X
            self.y = test_y 
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], device=device, dtype = torch.float32), torch.tensor(self.y[idx], device=device)


# Model 
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential (
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )


    def forward(self, x):
        return self.layers(x)
    

dataset = QM7_data()

trainloader = DataLoader(dataset, batch_size=1433*4, shuffle=True)
testloader = DataLoader(QM7_data(mode = 'test'), batch_size=1433, shuffle=False)

import torch.nn.functional as F
loss_function = F.mse_loss

mlp = MLP().to(device)


optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2)
for epoch in range(5000):
    current_loss = 0.0

    for i, data in enumerate(trainloader):
        inputs, targets = data
        targets = targets.view(-1,1)

        outputs = mlp(inputs)
        loss = loss_function(outputs, targets)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        current_loss += loss.item()

    print(f'Epoch {epoch+1} - Loss {current_loss/(i+1)}')
    torch.save(mlp.state_dict(), os.path.join('mlp.pt'))
    for data in testloader:
        inputs, targets = data
        targets = targets.view(-1,1)
        outputs = mlp(inputs)
    print(loss_function(outputs, targets)**0.5)


print("Training has completed")




