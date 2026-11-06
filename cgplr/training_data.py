import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T


def transforms(height=256, width=128, strong=False, training=True,
               color_jitter=0.2, grayscale=0.1, erasing=0.5, erasing_max=0.2):
    ops = [T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC)]
    if training:
        ops += [T.RandomHorizontalFlip(0.5), T.Pad(10), T.RandomCrop((height, width))]
    if strong:
        ops += [T.RandomApply([T.ColorJitter(color_jitter, color_jitter, color_jitter,
                                            min(color_jitter/2,0.5))], p=0.5),
                T.RandomGrayscale(p=grayscale)]
    ops += [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    if strong:
        ops += [T.RandomErasing(p=erasing, scale=(0.02, erasing_max), ratio=(0.3, 3.3), value=0)]
    return T.Compose(ops)


class Images(Dataset):
    def __init__(self, records, transform, strong_transform=None):
        self.records = records
        self.transform = transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path = self.records[index][0]
        with Image.open(path) as file:
            image = file.convert('RGB')
        weak = self.transform(image)
        if self.strong_transform is None:
            return weak, index
        return weak, self.strong_transform(image), index


class SelectiveBatchSampler(Sampler):

    def __init__(self, labels, deferred, batch_size, instances, iterations, deferred_fraction=0.25):
        self.groups = defaultdict(list)
        for index, label in enumerate(labels):
            if int(label) >= 0:
                self.groups[int(label)].append(index)
        self.deferred = [i for i, value in enumerate(deferred) if value]
        self.batch_size = batch_size
        self.instances = instances
        self.iterations = iterations
        self.deferred_count = int(batch_size * deferred_fraction)
        if batch_size < 2 or instances < 2 or iterations < 1 or not 0 <= deferred_fraction <= 1:
            raise ValueError('invalid batch size, instances, iterations or deferred fraction')
        if not self.groups and not self.deferred:
            raise ValueError('no eligible samples in this round')

    def __len__(self):
        return self.iterations

    def __iter__(self):
        ids = list(self.groups)
        for _ in range(self.iterations):
            n_def = self.deferred_count if self.deferred else 0
            if not ids:
                n_def = self.batch_size
            n_hard = self.batch_size - n_def
            batch = []
            while len(batch) < n_hard:
                take = min(self.instances, n_hard-len(batch))

                for pid in random.sample(ids, min(len(ids), max(1, (n_hard-len(batch))//take))):
                    batch.extend(random.choices(self.groups[pid], k=min(take, n_hard-len(batch))))
            if n_def:
                batch.extend(random.choices(self.deferred, k=n_def))
            random.shuffle(batch)
            yield batch
