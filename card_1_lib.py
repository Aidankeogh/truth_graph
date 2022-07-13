from torch.distributions.normal import Normal
import torch
from torch import tensor
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from third_party.gmm_torch.gmm import GaussianMixture

mu = torch.tensor(
    [[
        [1, 1],
        [-1, 1],
        [0, -1],
    ]]
)
var = torch.tensor(
    [[
        [0.1, 0.1],
        [0.1, 0.2],
        [0.3, 0.1],
    ]]
)

real_world = GaussianMixture(
    n_components=3,
    n_features=2,
    mu_init=mu,
    var_init=var,
    covariance_type="diag")

def visualize_gmm(gmm, n_samples=1000, samples=None, classes=None, show=True, **kde_kwargs):
    if samples is None:
        samples, classes = gmm.sample(n_samples)

    sample_df = pd.DataFrame({
        "x": samples[:, 0],
        "y": samples[:, 1],
        "class": classes
    })
    sns.kdeplot(data=sample_df, x="x", y="y", **kde_kwargs)
    if show:
        plt.show()