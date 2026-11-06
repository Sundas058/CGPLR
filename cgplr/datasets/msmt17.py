from __future__ import print_function, absolute_import
import os.path as osp

import glob
import re
from pathlib import Path
from ..utils.data import BaseImageDataset


def _process_dir(dir_path, relabel=False):
    img_paths = sorted(glob.glob(osp.join(dir_path, '*.jpg')))
    pattern = re.compile(r'([-\d]+)_c(\d+)')

    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
    dataset = []
    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1:
            continue  
        assert 1 <= camid <= 15
        camid -= 1  
        if relabel:
            pid = pid2label[pid]
        dataset.append((img_path, pid, camid))

    return dataset


class MSMT17(BaseImageDataset):
    dataset_dir = 'MSMT17_V1'

    def __init__(self, root, verbose=True, version='MSMT17_V1', combine_trainval=False, **kwargs):
        super(MSMT17, self).__init__()
        if version not in ('MSMT17_V1', 'MSMT17_V2'):
            raise ValueError('Unsupported MSMT17 version')
        self.dataset_dir = osp.join(root, version)
        base = Path(self.dataset_dir)
        if (base/'list_train.txt').is_file():
            train_folder, test_folder = ('train','test') if version == 'MSMT17_V1' else ('mask_train_v2','mask_test_v2')
            train = self._read_list(base/'list_train.txt', base/train_folder)
            if combine_trainval:
                train += self._read_list(base/'list_val.txt', base/train_folder)
            ids = {pid:i for i,pid in enumerate(sorted({r[1] for r in train}))}
            train = [(name,ids[pid],cam) for name,pid,cam in train]
            query = self._read_list(base/'list_query.txt', base/test_folder)
            gallery = self._read_list(base/'list_gallery.txt', base/test_folder)
        else:
            if combine_trainval:
                raise ValueError('train/val selection requires official MSMT17 list files')
            self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
            self.query_dir = osp.join(self.dataset_dir, 'query')
            self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
            self._check_before_run()
            train = _process_dir(self.train_dir, relabel=True)
            query = _process_dir(self.query_dir, relabel=False)
            gallery = _process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print('=> '+version+' loaded')
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    @staticmethod
    def _read_list(list_file, image_dir):
        records = []
        for line in list_file.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            relative, pid = line.rsplit(maxsplit=1)
            path = image_dir/relative
            if not path.is_file():
                raise FileNotFoundError(path)
            camera = int(path.name.split('_')[2])-1
            if not 0 <= camera < 15:
                raise ValueError('MSMT17 camera must be in 1..15')
            records.append((str(path.resolve()),int(pid),camera))
        if len({r[0] for r in records}) != len(records):
            raise ValueError('Duplicate MSMT17 list entries')
        return records

    def _check_before_run(self):

        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
