import lmdb
import os
import argparse
from tqdm import tqdm
import pickle as pkl
import mmengine


def parse_args():
    parser = argparse.ArgumentParser(description='Convert images to ldmb')
    parser.add_argument('image_root', help='directory which contains images')
    parser.add_argument('lmdb_path', help='path of generated lmdb file')
    parser.add_argument(
        '--img_ext',
        default='jpg',
        help='image name extensions such as jpg, png')
    args = parser.parse_args()
    return args


def img_to_lmdb(image_root, lmdb_path, img_ext):
    image_paths = [
        os.path.join(image_root, image_name)
        for image_name in os.listdir(image_root) if img_ext in image_name
    ]
    mmengine.mkdir_or_exist(lmdb_path)
    db = lmdb.open(lmdb_path, map_size=1099511627776)
    image_keys = []
    print(f'start writing {len(image_paths)} images to {lmdb_path}')
    with db.begin(write=True) as txn:
        for id, img_path in enumerate(tqdm(image_paths)):
            with open(img_path, 'rb') as f:
                image_data = f.read()
                key = str(id).zfill(8)
                image_keys.append(key)
                txn.put(key.encode(), image_data)
    db.close()
    meta_info = dict(file_name_list=image_keys)
    with open(os.path.join(lmdb_path, 'meta.pkl'), 'wb') as f:
        pkl.dump(meta_info, f)


def main():
    args = parse_args()
    img_to_lmdb(args.image_root, args.lmdb_path, args.img_ext)


if __name__ == '__main__':
    main()
