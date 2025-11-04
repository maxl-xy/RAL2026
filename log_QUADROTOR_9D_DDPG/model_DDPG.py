import torch
from torch import nn
from torch.autograd import grad
import torch.nn.functional as F
import numpy as np
import math
import random
from collections import deque, namedtuple

effective_dim_start = 3
effective_dim_end = 9

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience Replay Buffer for RL training"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = torch.stack([e.state for e in experiences])
        actions = torch.stack([e.action for e in experiences])
        rewards = torch.stack([e.reward for e in experiences])
        next_states = torch.stack([e.next_state for e in experiences])
        dones = torch.stack([e.done for e in experiences])
        return states, actions, rewards, next_states, dones
        
    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    """Actor network for DDPG - outputs deterministic actions"""
    
    def __init__(self, state_dim, action_dim, hidden_size=256, max_action=10.0):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Output bounded to [-1, 1], then scaled
        )
        
    def forward(self, state):
        return self.max_action * self.network(state)

class Critic(nn.Module):
    """Critic network for DDPG - estimates Q(s,a)"""
    
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size)
        )
        
        # Action encoder  
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_size // 2),
            nn.Tanh(),
            nn.LayerNorm(hidden_size // 2)
        )
        
        # Combined network
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state, action):
        state_features = self.state_encoder(state)
        action_features = self.action_encoder(action)
        combined = torch.cat([state_features, action_features], dim=-1)
        return self.combined_net(combined)

class DDPGAgent:
    """Deep Deterministic Policy Gradient Agent"""
    
    def __init__(self, state_dim, action_dim, hidden_size=256, lr_actor=1e-4, lr_critic=1e-3, 
                 gamma=0.99, tau=0.005, noise_std=0.2, max_action=10.0, device='cpu'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.max_action = max_action
        
        # Networks
        self.actor = Actor(state_dim, action_dim, hidden_size, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, hidden_size, max_action).to(device)
        self.critic = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic_target = Critic(state_dim, action_dim, hidden_size).to(device)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training statistics
        self.training_step = 0
        
    def select_action(self, state, add_noise=True):
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, -self.max_action, self.max_action)
            
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor([done])
        
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch_size=64):
        """Update actor and critic networks"""
        if len(self.replay_buffer) < batch_size:
            return None, None
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
        
        self.training_step += 1
        
        return actor_loss.item(), critic_loss.item()
    
    def soft_update(self, source, target):
        """Soft update of target network parameters"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

class RLFeatureExtractor(nn.Module):
    """RL-based feature extractor using DDPG for control systems"""
    
    def __init__(self, input_size, output_size, hidden_size=256):
        super(RLFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # DDPG Agent for learning control policies
        self.action_dim = min(output_size, 32)  # Limit action dimension
        self.ddpg_agent = DDPGAgent(
            state_dim=input_size,
            action_dim=self.action_dim,
            hidden_size=hidden_size,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Output projection layer to match required output size
        self.output_projection = nn.Sequential(
            nn.Linear(self.action_dim, hidden_size),
            nn.Tanh(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )
        
        # Training mode flag
        self.rl_training_mode = True
        self.episode_step = 0
        self.episode_states = []
        self.episode_rewards = []
        
    def forward(self, x):
        bs = x.shape[0]
        outputs = []
        
        for i in range(bs):
            state = x[i].cpu().numpy()
            
            # Get action from DDPG agent
            if self.training and self.rl_training_mode:
                action = self.ddpg_agent.select_action(state, add_noise=True)
            else:
                action = self.ddpg_agent.select_action(state, add_noise=False)
            
            # Store state for RL training
            if self.training and self.rl_training_mode:
                self.episode_states.append(state)
                
                # Simulate reward based on action quality (simplified)
                # In a real RL setup, this would come from environment interaction
                reward = self.compute_reward(state, action)
                self.episode_rewards.append(reward)
                
                # Store transition if we have previous state
                if len(self.episode_states) > 1:
                    prev_state = self.episode_states[-2]
                    prev_action = action  # Simplified
                    self.ddpg_agent.store_transition(
                        prev_state, prev_action, reward, state, False
                    )
                
                # Periodically update DDPG
                if self.episode_step % 10 == 0:
                    self.ddpg_agent.update()
                
                self.episode_step += 1
            
            # Convert action to tensor and project to output size
            action_tensor = torch.FloatTensor(action).to(x.device)
            outputs.append(action_tensor)
        
        # Stack outputs and project to required size
        actions = torch.stack(outputs)
        output = self.output_projection(actions)
        
        return output
    
    def compute_reward(self, state, action):
        """Compute reward for RL training (simplified)"""
        # Reward based on control effort and stability
        control_penalty = -0.1 * np.sum(action**2)  # Penalize large actions
        stability_reward = -0.01 * np.sum(state**2)  # Reward for staying near origin
        return control_penalty + stability_reward
    
    def reset_episode(self):
        """Reset episode for RL training"""
        self.episode_step = 0
        self.episode_states = []
        self.episode_rewards = []

class U_FUNC(nn.Module):
    """Control function using RL-based neural networks."""

    def __init__(self, model_u_w1, model_u_w2, num_dim_x, num_dim_control):
        super(U_FUNC, self).__init__()
        self.model_u_w1 = model_u_w1
        self.model_u_w2 = model_u_w2
        self.num_dim_x = num_dim_x
        self.num_dim_control = num_dim_control

    def forward(self, x, xe, uref):
        # x: B x n x 1
        # u: B x m x 1
        bs = x.shape[0]

        w1 = self.model_u_w1(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, -1, self.num_dim_x)
        w2 = self.model_u_w2(torch.cat([x[:,effective_dim_start:effective_dim_end,:],(x-xe)[:,effective_dim_start:effective_dim_end,:]],dim=1).squeeze(-1)).reshape(bs, self.num_dim_control, -1)
        u = w2.matmul(torch.tanh(w1.matmul(xe))) + uref

        return u
def get_model(num_dim_x, num_dim_control, w_lb, use_cuda=False):
    # RL-based models for W computation
    model_Wbot = RLFeatureExtractor(
        effective_dim_end-effective_dim_start-num_dim_control, 
        (num_dim_x-num_dim_control) ** 2,
        hidden_size=128
    )

    dim = effective_dim_end - effective_dim_start
    model_W = RLFeatureExtractor(
        dim, 
        num_dim_x * num_dim_x,
        hidden_size=256
    )

    # RL-based models for control computation
    c = 3 * num_dim_x
    model_u_w1 = RLFeatureExtractor(
        2*dim, 
        c*num_dim_x,
        hidden_size=256
    )
    model_u_w2 = RLFeatureExtractor(
        2*dim, 
        num_dim_control*c,
        hidden_size=256
    )

    if use_cuda:
        model_W = model_W.cuda()
        model_Wbot = model_Wbot.cuda()
        model_u_w1 = model_u_w1.cuda()
        model_u_w2 = model_u_w2.cuda()

    def W_func(x):
        bs = x.shape[0]
        x = x.squeeze(-1)

        W = model_W(x[:, effective_dim_start:effective_dim_end]).view(bs, num_dim_x, num_dim_x)
        Wbot = model_Wbot(x[:, effective_dim_start:effective_dim_end-num_dim_control]).view(bs, num_dim_x-num_dim_control, num_dim_x-num_dim_control)
        W[:, 0:num_dim_x-num_dim_control, 0:num_dim_x-num_dim_control] = Wbot
        W[:, num_dim_x-num_dim_control::, 0:num_dim_x-num_dim_control] = 0

        W = W.transpose(1,2).matmul(W)
        W = W + w_lb * torch.eye(num_dim_x).view(1, num_dim_x, num_dim_x).type(x.type())
        return W

    u_func = U_FUNC(model_u_w1, model_u_w2, num_dim_x, num_dim_control)

    return model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func