from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODONE: Formulate entropy term
        return torch.exp(self.log_alpha)

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODONE: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        obs = ptu.from_numpy(obs)
        action_dist = self(obs)
        if sample:
            action = action_dist.rsample()
        else:
            action = action_dist.mean.unsqueeze(0)
        return ptu.to_numpy(action.clip(self.action_range[0], self.action_range[1]))

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODONE: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        mean = self.mean_net(observation)
        clipped_logstd = self.logstd.clip(self.log_std_bounds[0], self.log_std_bounds[1])
        std = torch.exp(clipped_logstd).repeat(mean.shape[0], 1)
        action_distribution = sac_utils.SquashedNormal(mean, std)
        return action_distribution

    def update(self, obs, critic):
        # TODONE Update actor network and entropy regularizer
        # return losses and alpha value

        dist = self(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1, keepdim=True)
        q1, q2 = critic(obs, action)
        q = torch.min(q1, q2)
        actor_loss = ((self.alpha * log_prob).squeeze() - q).mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        alpha_loss = -(torch.exp(self.log_alpha) * (log_prob + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_loss, alpha_loss, self.alpha