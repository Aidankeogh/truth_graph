from random import sample
from lib.gmm_torch.gmm import GaussianMixture
from lib.visualization import plot_distribution
from lib.utils import sample_uniform, in_range
from typing import Dict
import torch
import pandas as pd


class MultiGMMEnv:
    def __init__(self, config: Dict):
        entities = []
        means = []
        vars = []
        pis = []

        for entity, cfg in config["entities"].items():
            for _ in range(cfg["n"]):
                dist = cfg["dist"]
                entities.append(entity)
                means.append(sample_uniform(dist["mean"]).unsqueeze(0))
                vars.append(sample_uniform(dist["var"]).unsqueeze(0))
                pis.append(sample_uniform(dist["pi"]).unsqueeze(0))
        means = torch.stack(means, dim=1)
        vars = torch.stack(vars, dim=1)
        pis = torch.softmax(torch.stack(pis, dim=1), 1)

        self.df = pd.DataFrame(
            {
                "entities": entities,
                "mean_x": means[0, :, 0],
                "mean_y": means[0, :, 1],
                "var_x": vars[0, :, 0],
                "var_y": vars[0, :, 1],
                "pi": pis[0, :, 0],
            }
        )

        self.gmm = GaussianMixture(
            n_components=means.shape[1],
            n_features=means.shape[2],
            mu_init=means,
            var_init=vars,
            pi_init=pis,
        )
        self.samples_per_step = config["samples_per_step"]
        self.view_radius = config["view_radius"]

    def step(self, locations):
        samples, components = self.gmm.sample(self.samples_per_step)
        sample_df = pd.DataFrame(
            {
                "entity": [self.df["entities"][idx.item()] for idx in components],
                "component": components,
                "x": samples[:, 0],
                "y": samples[:, 1],
            }
        )
        obs = {}
        for agent, location in locations.items():
            viewed = sample_df.apply(
                lambda col: in_range(location, [col["y"], col["y"]], self.view_radius),
                axis=1,
            )
            obs[agent] = sample_df[viewed]
        return obs
