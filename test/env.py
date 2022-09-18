from lib.env import MultiGMMEnv
import yaml
import torch

magic_forest_cfg = yaml.load(open("cfg/magic_forest.yaml"))

magic_forest = MultiGMMEnv(magic_forest_cfg)

locations = {"aidan": torch.tensor([1, 1])}

print(magic_forest.step(locations))

magic_forest.render()
