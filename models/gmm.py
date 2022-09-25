from random import sample
from models.gmm_torch.gmm import GaussianMixture
from lib.visualization import plot_distribution
from typing import Dict, List, Union
import torch
import pandas as pd
from lib.visualization import plot_distribution
import matplotlib.pyplot as plt
from lib.utils import sample_uniform


class GMM:
    def __init__(
        self,
        config: Dict = None,
        df: pd.DataFrame = None,
        means: List[float] = None,
        vars: List[float] = None,
        pis: List[float] = None,
        n_clusters: int = None,
        n_dims: int = 2,
    ):

        if config is not None:
            self.init_from_config_dict(config)
        elif df is not None:
            self.init_from_df(df)
        elif means is not None:
            self.init_from_lists(means, vars, pis)
        else:
            self.init_rand(n_clusters, n_dims)

        self.df = pd.DataFrame(
            {
                "mean_x": self.means[0, :, 0],
                "mean_y": self.means[0, :, 1],
                "var_x": self.vars[0, :, 0],
                "var_y": self.vars[0, :, 1],
                "pi": self.pis[0, :, 0],
            }
        )

        self.gmm = GaussianMixture(
            n_components=self.means.shape[1],
            n_features=self.means.shape[2],
            mu_init=self.means,
            var_init=self.vars,
            pi_init=self.pis,
        )

    def init_rand(self, n_clusters: int, n_dims: int = 2):
        self.gmm = GaussianMixture(
            n_components=n_clusters,
            n_features=n_dims,
        )
        self.means = self.gmm.mu
        self.vars = self.gmm.var
        self.pis = self.gmm.pi

    def init_from_lists(self, means: List, vars: List, pis: List = None):
        self.means = torch.tensor([means])
        self.vars = torch.tensor([vars])
        if pis is None:
            pis = [[1] for _ in vars]
        self.pis = torch.tensor([pis])

    def init_from_config_dict(self, config: Dict = None):
        entities = []
        means = []
        vars = []
        pis = []

        for entity, cfg in config.items():
            if type(cfg) is dict:
                for _ in range(cfg["n"]):
                    dist = cfg["dist"]
                    entities.append(entity)
                    means.append(sample_uniform(dist["mean"]).unsqueeze(0))
                    vars.append(sample_uniform(dist["var"]).unsqueeze(0))
                    pis.append(sample_uniform(dist["pi"]).unsqueeze(0))
            elif type(cfg) is list:
                for dist in cfg:
                    entities.append(entity)
                    means.append(torch.tensor(dist["mean"]).unsqueeze(0))
                    vars.append(torch.tensor(dist["var"]).unsqueeze(0))
                    pis.append(torch.tensor(dist["pi"]).unsqueeze(0).unsqueeze(0))

        self.entities = entities
        self.means = torch.stack(means, dim=1)
        self.vars = torch.stack(vars, dim=1)
        self.pis = torch.softmax(torch.stack(pis, dim=1), 1)

    def init_from_df(self, df: pd.DataFrame):
        self.df = df
        self.entities = list(df.entities)
        self.means = torch.stack(
            [torch.tensor(df["mean_x"]), torch.tensor(df["mean_y"])], axis=1
        ).unsqueeze(0)

        self.vars = torch.stack(
            [torch.tensor(df["var_x"]), torch.tensor(df["var_y"])], axis=1
        ).unsqueeze(0)

        self.pis = torch.tensor(df["pi"]).unsqueeze(1).unsqueeze(0)

    def sample(self, n: int) -> pd.DataFrame:
        samples, _ = self.gmm.sample(n)
        return samples

    def render(self, **kwargs):
        plot_distribution(self.gmm, **kwargs)

    def render_entities_individually(self):
        pi_orig = self.gmm.pi[0, :, 0].clone()
        unique_entites = set(self.entities)
        fig, axs = plt.subplots(1, len(unique_entites) + 1)
        fig.set_size_inches(10, 2)

        plot_distribution(self.gmm, ax=axs[0], cbar=False)
        axs[0].set_title("all")

        for entity, ax in zip(unique_entites, axs[1:]):
            mask = torch.tensor([int(e == entity) for e in self.entities])
            self.gmm.pi[0, :, 0] = pi_orig * mask
            plot_distribution(self.gmm, ax=ax, cbar=False)
            ax.set_title(entity)

        self.gmm.pi[0, :, 0] = pi_orig

    def fit(self, data: torch.tensor):
        self.gmm.fit(data)

        self.means = self.gmm.mu
        self.vars = self.gmm.var
        self.pis = self.gmm.pi

        self.df = pd.DataFrame(
            {
                "mean_x": self.means[0, :, 0],
                "mean_y": self.means[0, :, 1],
                "var_x": self.vars[0, :, 0],
                "var_y": self.vars[0, :, 1],
                "pi": self.pis[0, :, 0],
            }
        )

    def loglikelihood(self, data: torch.tensor):
        return self.gmm._score(data)
