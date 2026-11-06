import torch


def nearest_neighbors(features, neighbors, chunk_size=512):
    if int(neighbors) < 1 or int(chunk_size) < 1:
        raise ValueError('neighbors and chunk_size must be positive integers')
    n = len(features)
    k = min(int(neighbors), max(n - 1, 0))
    if k == 0:
        return torch.empty((n, 0), dtype=torch.long, device=features.device)
    result = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        similarities = features[start:end] @ features.T
        similarities[torch.arange(end-start, device=features.device),
                     torch.arange(start, end, device=features.device)] = -torch.inf
        result.append(similarities.topk(k, dim=1).indices)
    return torch.cat(result)
