import pickle

try:
    from utils import death_masking, vectorize_state
except ImportError: 
    from .utils import death_masking, vectorize_state


model = open(__file__[:-13] + f"/model.pkl", "rb")
predictor = pickle.load(model)


class PredatorAgent:
    def __init__(self):
        self.agent = predictor
    
    def act(self, state_dict):
        state = vectorize_state(state_dict)[:-10 * 3]
        return predictor.predict([state])[0][:2]


class PreyAgent:
    def __init__(self):
        self.agent = predictor
    
    def act(self, state_dict):
        state = vectorize_state(state_dict)[:-10 * 3]
        return predictor.predict([state])[0][2:]


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
        