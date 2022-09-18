import torch


def sample_uniform(cfg):

    min = torch.tensor(cfg["min"]).view(-1)
    max = torch.tensor(cfg["max"]).view(-1)
    return torch.rand(len(min)) * (max - min) + min  # wewrt


def in_range(viewer_position, target_position, viewer_range):
    viewer_position = torch.tensor(viewer_position).float().unsqueeze(0)
    target_position = torch.tensor(target_position).float().unsqueeze(0)
    dist = torch.cdist(viewer_position, target_position)
    return (dist <= viewer_range).item()
