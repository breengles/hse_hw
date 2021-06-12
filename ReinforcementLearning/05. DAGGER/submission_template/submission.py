import torch
import torch.nn as nn

try:
    from utils import death_masking, vectorize_state
except ImportError: 
    from .utils import death_masking, vectorize_state


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2 * 4 + 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
        )
    
    def forward(self, state):
        return torch.tanh(self.model(state))


model = Net()
model.load_state_dict(torch.load(__file__[:-13] + f"/model.pt"))
model.eval()


class PredatorAgent:
    def __init__(self):
        self.agent = model
    
    def act(self, state_dict):
        with torch.no_grad():
            state = torch.FloatTensor(vectorize_state(state_dict)[:-10 * 3])
            return self.agent(state).data.numpy()[:2]


class PreyAgent:
    def __init__(self):
        self.agent = model
    
    def act(self, state_dict):
        with torch.no_grad():
            state = torch.FloatTensor(vectorize_state(state_dict)[:-10 * 3])
            return self.agent(state).data.numpy()[2:]


if __name__ == "__main__":
    from predators_and_preys_env.env import PredatorsAndPreysEnv
    
    pred = PredatorAgent()
    prey = PreyAgent()
    env = PredatorsAndPreysEnv(render=True)
    
    
    for _ in range(25):
        state = env.reset()
        done = False
        while not done:
            pred_actions = pred.act(state)
            prey_actions = prey.act(state)
            state, _, done = env.step(pred_actions, prey_actions)
        