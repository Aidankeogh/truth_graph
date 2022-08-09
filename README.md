# Welcome to misinfo sim! 

This project is an attempt to simulate how misinformation spreads on social media so that we can find ways of making social media more robust to misinformation. 

Our goal is to find social media structures that make it easier to spread good ideas, and harder to spread bad ideas. 

The scope of this project is not just limited to political misinformation but rather attempts to mislead people online in general. One of the primary use cases being explored is as an ethical advertising platform. 

This project uses a [multi-agent reinforcement learning](https://en.wikipedia.org/wiki/Multi-agent_reinforcement_learning) setup similar to salesforce's [AI economist](https://www.salesforce.com/news/stories/introducing-the-ai-economist-why-salesforce-researchers-are-applying-machine-learning-to-economics/), except instead of using tax policy to optimize productivity/equality, we will use levers inside of social media systems to optimize our agents' understanding of their environment.

For nomenclature, misinformation means any information given with the intent of misleading the listener. For instance, in games like [mafia, warewolf, or among-us](https://en.wikipedia.org/wiki/Mafia_(party_game)), information from other players is potential misinformation because they might be on the other team, trying to trick you into forming a false belief.

We will call regular people **agents**, and any people trying to mislead **misinfo agents**. The false belief that a misinfo agent is trying to get an agent to believe is their **agenda**. 

Here is the high level structure of our sim
![Graphical overview](https://github.com/Aidankeogh/truth_graph/blob/master/misinfo-sim.png?raw=true)

I am also creating a developer log explaining this sim and how it works, targeted at people with little to no ML knowledge. 

Check out my [sprint 1 devlog](https://github.com/Aidankeogh/truth_graph/blob/master/sprint_1_wolf_forest.ipynb) to get started!

Or check out my [Kanban board](https://github.com/Aidankeogh/truth_graph/projects/1) to see what I'm working on now! 
