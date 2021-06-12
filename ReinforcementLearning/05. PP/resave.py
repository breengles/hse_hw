import torch
import sys

model = torch.load(sys.argv[-1])


for idx, agent in enumerate(model.pred_agents):
    torch.save(agent.actor.state_dict(), f"submission_template/pred{idx}.pt")

for idx, agent in enumerate(model.prey_agents):
    torch.save(agent.actor.state_dict(), f"submission_template/prey{idx}.pt")
