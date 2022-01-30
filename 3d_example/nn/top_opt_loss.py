import torch
import torch.nn as nn


class TopOptLoss(nn.Module):

    def __init__(self):
        super(TopOptLoss, self).__init__()

    def forward(self, nn_rho, psi_elem, volume_fraction, penal, psi_0, alpha):
        objective = torch.sum(torch.div(psi_elem, nn_rho**penal)) / psi_0
        vol_constraint = ((torch.mean(nn_rho) / volume_fraction) - 1.0)

        return objective + alpha * pow(vol_constraint, 2), vol_constraint
