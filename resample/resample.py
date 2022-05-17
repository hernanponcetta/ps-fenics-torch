import time
import os

from fenics import *
import numpy as np
import torch
import torch.optim as optim

from nn.top_optimizer_nn import TopOptimizerNN

from msh2xdmf.msh2xdmf import import_mesh
from utils.utils import create_mid_points, create_time_stamp

dim = 3

# Create time stamp
ts = create_time_stamp()

# Directory name for data
data_directory = "output/{ts}".format(ts=ts)

# Create file for saving the results
xdmf = XDMFFile("{data_directory}/density.xdmf".format(data_directory=data_directory))

# Import mesh, more info:
mesh, boundaries_mf, association_table = import_mesh(
    prefix='mesh_h',
    dim=dim,
    directory="./mesh",
)

# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"

# Get mid points for each cell, filter and move them to cpu/gpu
mid_points = create_mid_points(mesh, dim)
mid_points = torch.tensor(mid_points, requires_grad=True).float().to(device)

# Load model and set model state
top_optimizer = TopOptimizerNN(dim, neurons_per_layer=20, numbers_of_layers=10, use_softmax=True)
top_optimizer.load_state_dict(torch.load("./model/model.pt"))
top_optimizer.eval()

# Predict density for each cell
density_new_tt = top_optimizer(mid_points)

# Convert density to numpy array
density_new_np = density_new_tt.cpu().detach().numpy()

D = FunctionSpace(mesh, "DG", 0)
density = Function(D, name="density")

# Assign new density to function space and solve
density.vector()[:] = density_new_np

xdmf.write(density)