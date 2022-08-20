import torch
import numpy as np

@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    return x.div_(5.).tanh_().mul(5.)

class Normal:
    def __init__(self, mu, log_sigma):
        self.mu = torch.clamp(mu, -5, 5)
        log_sigma = torch.clamp(log_sigma, -5, 5)
        self.std = log_sigma.mul(0.5).exp()

    def sample(self):
        eps = self.mu.mul(0).normal_()
        z = eps.mul_(self.std).add_(self.mu)
        return z, eps

    @staticmethod
    def get_standard(bs, nz, device):
        zeros = torch.zeros(bs, nz).to(device)
        return Normal(zeros, zeros)

    def sample_given_eps(self, eps):
        return eps * self.std + self.mu

    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / self.std
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi) - torch.log(self.std)
        log_p = torch.sum(log_p, dim=-1)
        return log_p

    def kl(self, normal_dist):
        assert normal_dist.mu.shape == self.mu.shape
        term1 = (self.mu - normal_dist.mu) / normal_dist.std
        term2 = self.std / normal_dist.std
        loss = 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)
        loss = torch.sum(loss, dim=-1)
        return loss

    def set_device(self, cuda_id):
        self.mu = self.mu.to(cuda_id)
        self.std = self.std.to(cuda_id)
