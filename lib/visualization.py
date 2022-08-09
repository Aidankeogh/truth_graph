import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import ticker
from pyrsistent import l
from torch import tensor
from torch.distributions.normal import Normal

from lib.gmm_torch.gmm import GaussianMixture


def visualize_gmm(
    gmm,
    n_samples=1000,
    samples=None,
    classes=None,
    show=True,
    scatter=False,
    **kde_kwargs
):
    if samples is None:
        samples, classes = gmm.sample(n_samples)

    sample_df = pd.DataFrame({"x": samples[:, 0], "y": samples[:, 1], "class": classes})
    sns.kdeplot(data=sample_df, x="x", y="y", **kde_kwargs)
    if scatter:
        sns.scatterplot(data=sample_df, x="x", y="y")
    if show:
        plt.show()


def plot_distribution(
    distribution,
    xrange=[-2, 2],
    yrange=[-2, 2],
    dots=100,
    plot_type="contour",
    ax=None,
    cbar=True,
    title=None,
    **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(1)
    plt.xlim(xrange)
    plt.ylim(yrange)

    dots = 100
    x = torch.linspace(xrange[0], xrange[1], dots)
    y = torch.linspace(yrange[0], yrange[1], dots)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    tsr = torch.stack([xx.flatten(), yy.flatten()], axis=1)
    z = torch.exp(distribution.score_samples(tsr))

    zz = z.reshape(xx.shape)
    if plot_type == "contour":
        cs = ax.contourf(xx, yy, zz, cmap="coolwarm")
        if cbar:
            plt.colorbar(cs)
    elif plot_type == "wireframe":
        ax = plt.axes(projection="3d")
        w = ax.plot_wireframe(xx, yy, zz)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("PDF")

    if title:
        ax.set_title(title)


if __name__ == "__main__":
    mu = torch.tensor(
        [
            [
                [1, 1],
                [-1, 1],
                [0, -1],
            ]
        ]
    )
    var = torch.tensor(
        [
            [
                [0.3, 0.3],
                [0.3, 0.4],
                [0.5, 0.3],
            ]
        ]
    )
    pi = torch.tensor(
        [
            [
                [0.3],
                [0.3],
                [0.4],
            ]
        ]
    )

    wolf_dens = GaussianMixture(
        n_components=3,
        n_features=2,
        mu_init=mu,
        var_init=var,
        pi_init=pi,
        covariance_type="diag",
    )

    fig, axs = plt.subplots(1, 2)
    plot_distribution(wolf_dens, ax=axs[0], title=False, cbar=False)
    plot_distribution(wolf_dens, ax=axs[1], title=False, cbar=False)
    plt.show()
