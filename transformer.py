#%%
import torch
from torch import nn
import sklearn.datasets as skdatasets
import torch.nn.functional as F
# %%
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim):
        super(SimpleTransformer, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim, bias=False)
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=1, 
                                                      dim_feedforward=embed_dim, 
                                                      dropout=0.0, bias=False,
                                                      batch_first=True)
        self.linear2 = nn.Linear(embed_dim, output_dim, bias=False)
    def forward(self, x):
        x = self.linear(x)
        x = self.transformer(x) 
        x = self.linear2(x)
        return x

# %%
model = SimpleTransformer(4, 8, 3)

iris = skdatasets.load_iris()
data = torch.tensor(iris.data, dtype=torch.float32)
target = torch.tensor(iris.target, dtype=torch.long)
print("Mean and std of dataset:")
print(data.mean(dim=0), data.std(dim=0))
print("========")
data = (data - data.mean(dim=0)) / data.std(dim=0)

dataset = torch.utils.data.TensorDataset(data[:-20], target[:-20])
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
from tqdm import tqdm
for epoch in tqdm(range(50)):
    losses = []
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        x = x.unsqueeze(1)
        optimizer.zero_grad()
        y_pred = model(x)
        y_pred = torch.squeeze(y_pred)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())

        
    print(f'Epoch {epoch}, Loss: {sum(losses)/len(losses)}')
# %%
# get the accuracy
y_pred = model(data.unsqueeze(1).to(device))
y_pred = torch.argmax(torch.squeeze(y_pred), dim=1)
accuracy = (y_pred == target.to(device)).sum().item() / len(target)
print(f'Accuracy: {accuracy}')
# %%
import struct
import numpy as np
filename = "weights.bin"
def write_weights(model, filename):
    with open(filename, 'wb') as f:
        for name, param in model.named_parameters():
            data = param.data.cpu().numpy()
            f.write(data.T.tobytes())

write_weights(model, filename)
# %%