# Copyright (c) OpenMMLab. All rights reserved.
import hashlib


def md5sum(file_path) -> str:
    """get md5sum of input file_path.

    Args:
        file_path (str): an existing file path

    Returns:
        str: md5 string
    """
    with open(file_path, 'rb') as f:
        md5sum_str = hashlib.md5(f.read()).hexdigest()
    return md5sum_str
