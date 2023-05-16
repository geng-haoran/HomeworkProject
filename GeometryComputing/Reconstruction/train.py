import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from epic_ops.pointcloud import save_pc_to_ply
import numpy as np
import pandas as pd
import wandb

from pyntcloud import PyntCloud

def save_point_cloud_to_ply(points, colors, save_root = "outout", save_name = "output.ply"):
    
    cloud = PyntCloud(pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=np.hstack((points, colors)),
        columns=["x", "y", "z", "red", "green", "blue"]))

    cloud.to_file(save_root +  "/" + save_name)
    

def read_xyz_file(filename):
    points = []
    normals = []
    with open(filename, 'r') as f:
        for line in f:
            x, y, z, nx, ny, nz = map(float, line.split())
            points.append((x, y, z))
            normals.append((nx, ny, nz))
    return np.array(points), np.array(normals)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.layers(x)

wandb.init(project='GeometryComputing')
DATA_ROOT = "data/gargoyle.xyz"
points, normals = read_xyz_file(DATA_ROOT)
# import pdb; pdb.set_trace()
# save_point_cloud_to_ply(points, points, save_root = ".", save_name = "output.ply")

# Convert to tensors
points = torch.from_numpy(points).float()
normals = torch.from_numpy(normals).float()

# Create the model
model = MLP()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
for epoch in range(10000):  # you may need to adjust the number of epochs
    optimizer.zero_grad()
    outputs = model(points)
    loss = criterion(outputs, normals)
    loss.backward()
    optimizer.step()
    wandb.log({"loss": loss.item()})
    if epoch % 1000 == 0:
        np.save(f"output/{epoch}_pred.npy", outputs.detach().numpy(), )
    print('Epoch [%d/100], Loss: %.4f' % (epoch+1, loss.item()))

