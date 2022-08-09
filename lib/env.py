from random import random

import matplotlib.pyplot as plt

# import pandas as pd
# import seaborn as sns
import torch
from torch import tensor

# from torch.distributions.normal import Normal

from lib.gmm_torch.gmm import GaussianMixture
from lib.visualization import plot_distribution


def create_gmm(mean, var):
    mean = tensor(mean).unsqueeze(0)
    var = tensor(var).unsqueeze(0)
    return GaussianMixture(
        n_components=mean.shape[1], n_features=mean.shape[2], mu_init=mean, var_init=var
    )


wolf_forest = create_gmm(
    mean=[
        [1, 1],  # Location of den 1
        [-1, 1],  # Location of den 2
        [1, -1],  # Location of den 3
    ],
    var=[
        [0.3, 0.3],
        [0.3, 0.4],
        [0.5, 0.3],
    ],
)

fox_den = create_gmm(
    mean=[
        [
            -1,
            -1,
        ],  # 2 dens overlapped for now since gaussianMixture throws an error with just 1 gaussian *shrug*
        # [-1, -1]  # Should behave identically to 1 gaussian. TODO: Fix this...
    ],
    var=[
        [0.2, 0.2],
        # [0.2, 0.2],
    ],
)


class Explorer:
    def __init__(self, environment=wolf_forest, name="dora"):
        self.environment = environment
        self.name = name
        self.wolf_sightings = torch.empty(0, 2)
        self.model = None
        self.notes = {}

    def sample(self, num_samples):
        sightings, _ = self.environment.sample(num_samples)
        return sightings

    def train(self, data):
        model = GaussianMixture(n_components=3, n_features=2)
        model.fit(data)  # Good ol model.fit
        return model


n_explorers = 4
wolves_per_day = 10
days = 5
explorer_names = ["dora", "boots", "tico", "benny"]

note_finding_probability = 0.5
bullshit_threshold = -4

found_note = (
    lambda: random() < note_finding_probability
)  # Returns true 50% of the time.

explorers = [Explorer(name=explorer_names[i]) for i in range(n_explorers)]
fox_notes = torch.empty(0, 2)
for day in range(days):
    print(f"Day {day + 1}.")

    # Code for plotting graphs next to eachother
    fig, axs = plt.subplots(1, n_explorers)
    fig.set_size_inches(10, 2)

    for explorer in explorers:  # Do regular exploration just like before.
        new_sightings = explorer.sample(wolves_per_day)
        explorer.wolf_sightings = torch.cat([explorer.wolf_sightings, new_sightings])

    fake_sightings, _ = fox_den.sample(wolves_per_day)
    fox_notes = torch.cat([fox_notes, fake_sightings])

    for explorer, ax in zip(explorers, axs):
        # Create a dictionary of all the notes an explorer finds on a given day
        notes_found = {
            other_explorer.name: other_explorer.wolf_sightings.clone()
            for other_explorer in explorers
            if other_explorer is not explorer and found_note()
        }
        if found_note():
            notes_found["swiper"] = fox_notes

        explorer.notes.update(notes_found)
        trusted_model = explorer.train(explorer.wolf_sightings)
        trusted_notes = {
            k: v
            for k, v in explorer.notes.items()
            if trusted_model._score(v) > bullshit_threshold
        }
        trusted_locations = torch.cat(
            [explorer.wolf_sightings] + list(trusted_notes.values())
        )

        explorer.model = explorer.train(trusted_locations)

        plot_distribution(explorer.model, ax=ax, title=explorer.name, cbar=False)

    plt.show()


# Define explorers, environment, additional info sources.
# social_media = SM()

# Self.info_dict = {}
# def step():
# experiences = explorers.sample(environment)
# posts = explorers.
#
# explorers.post(social_media)
# misinfo_agent.post(social_media)
# social_media.step()
#
# anecdotes = explorers.view(social_media)
#
# explorers.train(experiences, anecdotes)

explorers = [Explorer(name=explorer_names[i]) for i in range(n_explorers)]  # Reset
fox_notes = torch.empty(0, 2)  # Reset
for day in range(days):  # Main loop
    print(f"Day {day + 1}.")

    fig, axs = plt.subplots(1, n_explorers)  # Env.render()
    fig.set_size_inches(10, 2)  # Env.render()

    for explorer in explorers:  # in Env.step()
        new_sightings = explorer.sample(wolves_per_day)  # in Env.step()
        explorer.wolf_sightings = torch.cat(
            [explorer.wolf_sightings, new_sightings]
        )  # in Env.step()

    fake_sightings, _ = fox_den.sample(wolves_per_day)  # in Env.step()
    fox_notes = torch.cat([fox_notes, fake_sightings])  # in Env.step()

    for explorer, ax in zip(explorers, axs):
        # Create a dictionary of all the notes an explorer finds on a given day
        notes_found = {
            other_explorer.name: other_explorer.wolf_sightings.clone()
            for other_explorer in explorers
            if other_explorer is not explorer and found_note()
        }
        if found_note():
            notes_found["swiper"] = fox_notes

        explorer.notes.update(notes_found)
        trusted_model = explorer.train(explorer.wolf_sightings)
        trusted_notes = {
            k: v
            for k, v in explorer.notes.items()
            if trusted_model._score(v) > bullshit_threshold
        }
        trusted_locations = torch.cat(
            [explorer.wolf_sightings] + list(trusted_notes.values())
        )

        explorer.model = explorer.train(trusted_locations)

        plot_distribution(explorer.model, ax=ax, title=explorer.name, cbar=False)

    plt.show()
