import torch

try:
    from utils import vectorize_state
    from model import Net
except ImportError: 
    from .utils import vectorize_state
    from .model import Net


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
    
    model = Net()
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    
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
else:
    model = Net()
    model.load_state_dict(torch.load(__file__[:-13] + "/model.pt"))
    model.eval()
        