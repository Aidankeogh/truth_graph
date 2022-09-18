from random import sample
from lib.gmm_torch.gmm import GaussianMixture
from lib.visualization import plot_distribution
from lib.utils import in_range
from typing import Dict
import torch
import pandas as pd
from lib.visualization import plot_distribution
import matplotlib.pyplot as plt
from models.gmm import GMMModel


class MultiGMMEnv:
    def __init__(self, config: Dict):
        self.model = GMMModel(config["entities"])
        self.samples_per_step = config["samples_per_step"]
        self.view_radius = config["view_radius"]

    def step(self, agent_locations: Dict):
        sample_df = self.model.sample(self.samples_per_step)
        obs = {}
        for agent, location in agent_locations.items():
            viewed = sample_df.apply(
                lambda col: in_range(location, [col["x"], col["y"]], self.view_radius),
                axis=1,
            )
            obs[agent] = sample_df[viewed]
        return obs

    def render(self):
        self.model.render()
