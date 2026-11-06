from pathlib import Path
import numpy as np
from scipy.io import loadmat
from ..utils.data import BaseImageDataset


class CUHK03NPD(BaseImageDataset):
    def __init__(self, root, verbose=True, **kwargs):
        root = Path(root)
        config = root / 'cuhk03_new_protocol_config_detected.mat'
        image_dir = root / 'images_detected'
        if not config.is_file() or not image_dir.is_dir():
            raise FileNotFoundError('CUHK03-NP-D needs images_detected/ and ' + str(config))
        data = loadmat(config)
        ids = data['labels'].reshape(-1)
        names = data['filelist'].reshape(-1)
        indices = {part: data[part + '_idx'].reshape(-1).astype(int)-1
                   for part in ('train', 'query', 'gallery')}
        test_ids = set(ids[indices['query']]) | set(ids[indices['gallery']])
        if set(ids[indices['train']]) & test_ids:
            raise ValueError('CUHK03 protocol has overlapping train/test identities')
        if not set(ids[indices['query']]).issubset(set(ids[indices['gallery']])):
            raise ValueError('CUHK03 query identities must have gallery matches')
        train_ids = sorted(set(ids[indices['train']].tolist()))
        if len(train_ids) != 767 or len(set(ids[indices['gallery']])) != 700:
            raise ValueError('Expected the official 767/700 new protocol')
        relabel = {pid: i for i, pid in enumerate(train_ids)}
        for part, selected in indices.items():
            records = []
            for index in selected:
                name = names[index]
                while isinstance(name, np.ndarray):
                    name = name.item()
                name = str(name)
                path = image_dir / name
                if not path.is_file():
                    raise FileNotFoundError(path)
                pid = relabel[ids[index]] if part == 'train' else int(ids[index])
                camera = int(name.split('_')[2]) - 1
                records.append((str(path.resolve()), pid, camera))
            setattr(self, part, sorted(records))
            info = self.get_imagedata_info(records)
            for suffix, value in zip(('pids', 'imgs', 'cams'), info):
                setattr(self, 'num_' + part + '_' + suffix, value)
        if verbose:
            self.print_dataset_statistics(self.train, self.query, self.gallery)
