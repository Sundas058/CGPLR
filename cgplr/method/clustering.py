import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import DBSCAN


def valid_cluster_mask(pseudo_labels, outlier_label=-1):

    labels = torch.as_tensor(pseudo_labels)
    return labels != outlier_label


def compute_cluster_centroids(features, pseudo_labels, outlier_label=-1):

    features = F.normalize(torch.as_tensor(features).float(), dim=1)
    labels = torch.as_tensor(pseudo_labels, device=features.device).long()
    cluster_ids = torch.unique(labels[labels != outlier_label], sorted=True)

    if cluster_ids.numel() == 0:
        empty = features.new_empty((0, features.size(1)))
        return cluster_ids, empty

    centroids = []
    for cluster_id in cluster_ids:
        members = features[labels == cluster_id]
        centroids.append(members.mean(dim=0))

    centroids = F.normalize(torch.stack(centroids, dim=0), dim=1)
    return cluster_ids, centroids


def compact_cluster_labels(pseudo_labels, outlier_label=-1):

    labels = torch.as_tensor(pseudo_labels).long().clone()
    compact = labels.new_full(labels.size(), outlier_label)
    valid_ids = torch.unique(labels[labels != outlier_label], sorted=True)
    mapping = {}
    for new_id, old_id in enumerate(valid_ids.tolist()):
        compact[labels == int(old_id)] = int(new_id)
        mapping[int(old_id)] = int(new_id)
    return compact, mapping


def cluster_with_dbscan(distance_matrix, eps=0.55, min_samples=4, n_jobs=-1):

    distances = np.asarray(distance_matrix)
    clusterer = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='precomputed',
        n_jobs=n_jobs,
    )
    return clusterer.fit_predict(distances)
