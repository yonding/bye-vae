import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def get_vae(args):
    vae = Autoencoder(args.D_in, args.H1, args.H2, args.H3, args.latent_dim).to(args.device)
    args.n_parameters = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print("------------ NUMBER OF PARAMETERS ------------")
    print(args.n_parameters)
    print("----------------------------------------------")
    print()
    vae.apply(weights_init_uniform_rule)
    return vae

class Autoencoder(nn.Module):
    def __init__(self, D_in, H1, H2, H3, latent_dim):
        # Encoder
        super(Autoencoder, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.lin_bn1 = nn.BatchNorm1d(num_features=H1)  # Batch normalization
        self.linear2 = nn.Linear(H1, H2)
        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)  # Batch normalization
        self.linear3 = nn.Linear(H2, H3)
        self.lin_bn3 = nn.BatchNorm1d(num_features=H3)  # Batch normalization
        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(H3, latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)
        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, H3)
        self.fc_bn4 = nn.BatchNorm1d(H3)  # Batch normalization
        # Decoder
        self.linear4 = nn.Linear(H3, H2)
        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)
        self.linear5 = nn.Linear(H2, H1)
        self.lin_bn5 = nn.BatchNorm1d(num_features=H1)
        self.linear6 = nn.Linear(H1, D_in)
        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)
        self.relu = nn.ReLU()  # Activation Function

    def encode(self, x):
        lin1 = self.relu(self.lin_bn1(self.linear1(x)))
        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))
        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))
        fc1 = F.relu(self.bn1(self.fc1(lin3)))
        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        return r1, r2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))
        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))
        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))
        return self.lin_bn6(self.linear6(lin5))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    


class customLoss(nn.Module):
    def __init__(self, args):
        super(customLoss, self).__init__()
        self.args = args
        self.mse_loss = nn.MSELoss(reduction="sum")

    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar
    def forward(self, x_recon, z, mu, logvar):
        loss_MSE = self.mse_loss(x_recon, z)
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss_MSE + self.args.kld_rate * loss_KLD

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find("Linear") != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)