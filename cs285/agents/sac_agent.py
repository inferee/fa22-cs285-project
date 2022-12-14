from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.sac_utils import soft_update_params
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent
import gym
from cs285.policies.sac_policy import MLPPolicySAC
from cs285.critics.sac_critic import SACCritic
import cs285.infrastructure.pytorch_util as ptu
import torch

class SACAgent(BaseAgent):
    def __init__(self, env: gym.Env, agent_params):
        super(SACAgent, self).__init__()

        self.env = env
        self.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.critic_tau = 0.005
        self.learning_rate = self.agent_params['learning_rate']

        self.cds_percentile = self.agent_params['cds_percentile']
        self.cds_gamma = self.agent_params['cds_gamma']
        self.cds_threshold = 0
        self.soft_cds_temp = 1
        self.soft_cds_reweight = 1

        self.actor = MLPPolicySAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
            action_range=self.action_range,
            init_temperature=self.agent_params['init_temperature']
        )
        self.actor_update_frequency = self.agent_params['actor_update_frequency']
        self.critic_target_update_frequency = self.agent_params['critic_target_update_frequency']

        self.critic = SACCritic(self.agent_params)
        self.critic_target = copy.deepcopy(self.critic).to(ptu.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.training_step = 0
        self.replay_buffer = ReplayBuffer(max_size=100000)

    def update_critic(self, ob_no, ac_na, next_ob_no, re_n, terminal_n, weights_n):
        # TODONE: 
        # 1. Compute the target Q value. 
        # HINT: You need to use the entropy term (alpha)
        # 2. Get current Q estimates and calculate critic loss
        # 3. Optimize the critic 
        next_dist = self.actor(next_ob_no)
        next_ac_na = next_dist.rsample()
        log_prob = next_dist.log_prob(next_ac_na).sum(-1, keepdim=True)
        q1_target, q2_target = self.critic_target(next_ob_no, next_ac_na)
        q_target = re_n + self.gamma * (1 - terminal_n) * (torch.min(q1_target, q2_target) - (self.actor.alpha * log_prob).squeeze())

        q1, q2 = self.critic(ob_no, ac_na)
        critic_loss = torch.sum(weights_n * (q1 - q_target)**2 + weights_n * (q2 - q_target) ** 2)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        return critic_loss

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, is_original = None):
        # TODONE 
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)
        re_n = ptu.from_numpy(re_n)
        next_ob_no = ptu.from_numpy(next_ob_no)
        terminal_n = ptu.from_numpy(terminal_n)
        if is_original is None:
            weights_n = torch.ones(ob_no.size()[0], device = ptu.device) / ob_no.size()[0]
        else:
            is_original = torch.from_numpy(is_original).to(ptu.device)
            weights_n = self.get_soft_cds_weights(ob_no, ac_na, is_original) / (self.soft_cds_reweight * ob_no.size()[0])

        # 1. Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            critic_loss = self.update_critic(ob_no, ac_na, next_ob_no, re_n, terminal_n, weights_n)

        # 2. Softly update the target every critic_target_update_frequency (HINT: look at sac_utils)
        if self.training_step % self.critic_target_update_frequency == 0:
            soft_update_params(self.critic, self.critic_target, self.critic_tau)

        # 3. Implement following pseudocode:
        # If you need to update actor
        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            actor_loss, alpha_loss, alpha = self.actor.update(ob_no, self.critic)

        self.training_step += 1
        # 4. gather losses for logging
        loss = OrderedDict()
        loss['Critic_Loss'] = critic_loss
        loss['Actor_Loss'] = actor_loss
        loss['Alpha_Loss'] = alpha_loss
        loss['Temperature'] = alpha

        loss['CDS Threshold'] = self.cds_threshold
        if is_original is not None:
            loss['CDS Reweight'] = self.soft_cds_reweight
            loss['CDS Temp'] = self.soft_cds_temp

        return loss

    def get_soft_cds_weights(self, ob_no, ac_na, is_original):
        weights = torch.ones(ob_no.size()[0], device=ptu.device)
        with torch.no_grad():
            q1, q2 = self.critic(ob_no, ac_na)
            q = torch.min(q1, q2)
            q = q.flatten()
            delta = (q - self.cds_threshold)
            sigmoid = 1.0 / (1.0 + torch.exp(-delta / self.soft_cds_temp))
            weights[~is_original] = sigmoid[~is_original]
        self.soft_cds_reweight = self.cds_gamma * self.soft_cds_reweight + (1 - self.cds_gamma) * torch.sum(weights) / weights.size()[0]
        return weights

    def update_cds_threshold(self, ob_no, ac_na):
        """Update the data sharing percentile with an exponentially moving average"""
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na)

        with torch.no_grad():
            q1, q2 = self.critic(ob_no, ac_na)
            q = torch.min(q1, q2)
            k = max(int(self.cds_percentile * len(q)), 1)
            threshold = torch.kthvalue(q, k).values
            if self.cds_threshold is None:
                self.cds_threshold = threshold
            else:
                self.cds_threshold = self.cds_gamma * self.cds_threshold + (1 - self.cds_gamma) * threshold
            self.soft_cds_temp = self.cds_gamma * self.soft_cds_temp + (1 - self.cds_gamma) * torch.std(q).item()

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def add_conservative_soft(self, paths):
        self.replay_buffer.add_rollouts(paths, is_original = False)

    def add_conservative_hard(self, paths):
        observations, actions, next_observations, terminals, concatenated_rews, _ = convert_listofrollouts(paths)
        with torch.no_grad():
            q1, q2 = self.critic(ptu.from_numpy(observations), ptu.from_numpy(actions))
            q = torch.min(q1, q2)
            share_locs = q >= self.cds_threshold
            if not share_locs.any():
                return
        
        share_locs = ptu.to_numpy(share_locs)
        if share_locs.ndim == 0:
            share_locs = np.array([share_locs])

        observations = observations[share_locs]
        actions = actions[share_locs]
        next_observations = next_observations[share_locs]
        terminals = terminals[share_locs]
        concatenated_rews = concatenated_rews[share_locs]

        self.replay_buffer.add_transitions(observations, actions, next_observations, terminals, concatenated_rews)

    def sample(self, batch_size, get_is_original = False, orig_data_only = False):
        return self.replay_buffer.sample_random_data(batch_size, get_is_original = get_is_original, orig_data_only = orig_data_only)
