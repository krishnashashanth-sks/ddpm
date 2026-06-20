import torch
import torch.nn.functional as F
import torch.nn as nn

class DenoisingScoreMatchingLoss(nn.Module):
  def __init__(self,sde,eps=1e-5):
    super().__init__()
    self.sde=sde
    self.eps=eps
  def forward(self,score_model,x0):
    batch_size=x0.shape[0]
    t=torch.rand(batch_size,device=x0.device)*(self.sde.T-self.eps)+self.eps
    xt,std,true_noise=self.sde.marginal_sde(x0,t)
    predicted_noise=score_model(xt,t)
    loss=F.mse_loss(predicted_noise,true_noise,reduction='mean')
    return loss