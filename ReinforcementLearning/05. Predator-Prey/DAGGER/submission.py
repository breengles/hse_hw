import torch
import numpy as np
try:
    from utils import vectorize_state
    from model import Net, SinglePred, SinglePrey
except ImportError: 
    from .utils import vectorize_state
    from .model import Net, SinglePred, SinglePrey


class PredatorAgent:
    def __init__(self):
        self.agent = pred_model
    
    def act(self, state_dict):
        with torch.no_grad():
            actions = []
            state = np.array(vectorize_state(state_dict)[:-10 * 3])
            for i in range(2):
                idxes = list(range(i * 4, (i + 1) * 4)) \
                    + list(range(2 * 4, 2 * 4 + 5 * 5))
                s = torch.FloatTensor([state[idxes]])
                actions.append(self.agent(s).numpy().item())
            return actions
        # return self.agent(state).data.numpy()[:2]


class PreyAgent:
    def __init__(self):
        self.agent = prey_model
        self.preds_ids = list(range(0, 2 * 4))
    
    def act(self, state_dict):
        with torch.no_grad():
            actions = []
            state = np.array(vectorize_state(state_dict)[:-10 * 3])
            for i in range(5):
                prey = list(range(2 * 4 + i * 5, 2 * 4 + (i + 1) * 5))
                idxes = self.preds_ids + prey
                s = torch.FloatTensor([state[idxes]])
                actions.append(self.agent(s).numpy().item())
            return actions
        # return self.agent(state).data.numpy()[2:]


if __name__ == "__main__":
    from predators_and_preys_env.env import PredatorsAndPreysEnv
    
    # model = Net()
    # model.load_state_dict(torch.load("model.pt"))
    # model.eval()
    
    pred_model = SinglePred()
    prey_model = SinglePrey()
    pred_model.load_state_dict(torch.load("pred_bn_200.pt", map_location="cpu"))
    prey_model.load_state_dict(torch.load("prey_bn_339.pt", map_location="cpu"))
    pred_model.eval()
    prey_model.eval()
    
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
    # model = Net()
    # model.load_state_dict(torch.load(__file__[:-13] + "/model.pt"))
    # model.eval()
    
    pred_model = SinglePred()
    prey_model = SinglePrey()
    pred_model.load_state_dict(torch.load(__file__[:-13] + "/pred_bn_200.pt"))
    prey_model.load_state_dict(torch.load(__file__[:-13] + "/prey_bn_200.pt"))
    pred_model.eval().to("cpu")
    prey_model.eval().to("cpu")
        