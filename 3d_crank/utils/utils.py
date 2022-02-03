from datetime import datetime

import numpy as np
import torch
from fenics import cells


# Helper function to obtain a unix timestamp
def create_time_stamp():
    return datetime.timestamp(datetime.now())


# Helper function to extract mid points for each cell
def create_mid_points(mesh, dim):
    mid_points = [cell.midpoint().array()[:] for cell in cells(mesh)]
    mid_points = np.array([row[:dim] for row in mid_points])
    mid_points = torch.tensor(mid_points, requires_grad=True).float()
    return mid_points