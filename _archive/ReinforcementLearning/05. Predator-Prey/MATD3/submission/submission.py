import torch
import torch.nn as nn

try:
    from utils import death_masking, relative_agents_states
except ImportError: 
    from .utils import death_masking, relative_agents_states


STATE_DIM = 2 * 4 + 5 * 5 + 10 * 3


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64, temperature=1):
        super().__init__()
        self.temperature = temperature
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            # nn.LayerNorm(action_dim),
        )
        self.model[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, return_raw=False):
        out = self.model(state) / self.temperature
        # print(out)
        if return_raw:
            return out, torch.tanh(out)
        else:
            return torch.tanh(out)


class PredatorAgent:
    def __init__(self):
        # self.agents = [agent.actor for agent in maddpg.pred_agents]
        self.agents = [Actor(STATE_DIM, 1, temperature=30), Actor(STATE_DIM, 1, temperature=30)]
        for idx, agent in enumerate(self.agents):
            agent.load_state_dict(torch.load(__file__[:-13] + f"/pred{idx}.pt", map_location="cpu"))
    
    def act(self, state_dict):
        state = relative_agents_states(death_masking(state_dict))
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device="cpu")
            actions = [agent(state[idx]).squeeze(-1).numpy().item() for idx, agent in enumerate(self.agents)]
            
        return actions


class PreyAgent:
    def __init__(self):
        self.agents = [Actor(STATE_DIM, 1, temperature=30) for _ in range(5)]
        for idx, agent in enumerate(self.agents):
            agent.load_state_dict(torch.load(__file__[:-13] + f"/prey{idx}.pt", map_location="cpu"))
    
    def act(self, state_dict):
        state = relative_agents_states(death_masking(state_dict))
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float, device="cpu")
            actions = [agent(state[2 + idx]).squeeze(-1).numpy().item() for idx, agent in enumerate(self.agents)]
            
        return actions


if __name__ == "__main__":
    from predators_and_preys_env.env import PredatorsAndPreysEnv
    
    pred = PredatorAgent()
    prey = PreyAgent()
    env = PredatorsAndPreysEnv(render=True)
    
    state = env.reset()
    print(pred.act(state), prey.act(state))
    
    
    