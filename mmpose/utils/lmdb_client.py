import lmdb
import numpy as np
import mmcv


class LmdbClient:

    def __init__(self, color_type='grayscale') -> None:
        self.lmdb_map = dict()
        self.color_type = color_type

    def get(self, image_info: str):
        lmdb_path, image_id = image_info.split(':')
        if lmdb_path not in self.lmdb_map:
            lmdb_env = lmdb.open(
                lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)
            lmdb_txn = lmdb_env.begin()
            self.lmdb_map[lmdb_path] = dict(env=lmdb_env, txn=lmdb_txn)
        img_array = self.lmdb_map[lmdb_path]['txn'].get(image_id.encode())
        img = mmcv.imfrombytes(img_array, flag=self.color_type, backend='cv2')
        if self.color_type == 'grayscale':
            img = img[:, :, np.newaxis]
        return img

    def __del__(self):
        for _, v in self.lmdb_map.items():
            v['env'].close()
