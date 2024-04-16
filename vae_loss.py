import torch
import torch.nn as nn

class customLoss(nn.Module):
    def __init__(self, args):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")
        self.args = args

    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar
    def forward(self, x_recon, z, mu, logvar, mse_loss):
        loss_MSE = self.mse_loss(x_recon, z)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        mse_loss[0]+=loss_MSE
        return loss_MSE + self.args.kld_rate * loss_KLD
