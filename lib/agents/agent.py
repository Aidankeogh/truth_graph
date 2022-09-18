import coolname
from models.gmm import GMMModel


class BaseAgent:
    def __init__(self) -> None:
        raise NotImplementedError


class GMMAgent(BaseAgent):
    def __init__(self, model_cfg, name=None):
        self.name = name
        self.model = GMMModel(model_cfg)

    def sample(self, num_samples):
        sightings, _ = self.environment.sample(num_samples)
        return sightings

    def train(self, data):
        self.samplemodel.fit(data)  # Good ol model.fit
        return model
