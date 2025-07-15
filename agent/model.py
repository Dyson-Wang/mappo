import torch.nn as nn
import torch


class Actor(nn.Module):
    def __init__(self, num_states, num_points, num_channels, pmax):
        super(Actor, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(num_states, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.point_header = nn.Sequential(nn.Linear(128, 128),
                                          nn.ReLU(),
                                          nn.Linear(128, num_points),
                                          nn.Softmax(dim=-1))
        self.channel_header = nn.Sequential(nn.Linear(128, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, num_channels),
                                            nn.Softmax(dim=-1))
        self.power_mu_header = nn.Sequential(nn.Linear(128, 64),
                                             nn.ReLU(),
                                             nn.Linear(64, 1),
                                             nn.Sigmoid())
        self.power_sigma_header = nn.Sequential(nn.Linear(128, 64),
                                                nn.ReLU(),
                                                nn.Linear(64, 1),
                                                nn.Softplus())
        self.pmax = pmax

    def forward(self, x):
        code = self.base(x)
        prob_points = self.point_header(code)
        prob_channels = self.channel_header(code)
        power_mu = self.power_mu_header(code) * (self.pmax - 1e-10) + 1e-10
        power_sigma = self.power_sigma_header(code)
        return prob_points, prob_channels, (power_mu, power_sigma)


class Critic(nn.Module):
    def __init__(self, num_states, num_agents):
        super(Critic, self).__init__()
        # self.embed_dim = 32
        # self.seq_len = num_states
        self.num_agents = num_agents

        # self.embedding = nn.Linear(num_states, self.seq_len * self.embed_dim)
        # self.attn = nn.MultiheadAttention(self.embed_dim, num_heads=2, batch_first=True)
        self.mlp = nn.Sequential(
            # nn.Linear(self.seq_len * self.embed_dim, 256),
            nn.Linear(num_states, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents)
        )

    def forward(self, x):
        # x: (batch, num_states) 或 (1, num_states)
        # if x.dim() == 1:
        #     x = x.unsqueeze(0)
        # batch_size = x.size(0)
        # embed = self.embedding(x)  # (batch, seq_len * embed_dim)
        # embed = embed.view(batch_size, self.seq_len, self.embed_dim)  # (batch, seq_len, embed_dim)
        # attn_out, _ = self.attn(embed, embed, embed)  # (batch, seq_len, embed_dim)
        # attn_flat = attn_out.reshape(batch_size, -1)  # (batch, seq_len * embed_dim)
        return self.mlp(x)
