# Copyright (c) OpenMMLab. All rights reserved.
import logging


def init_log(log_file='test.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置打印级别
    # formatter = (logging.Formatter('%(asctime)s %(filename)s %(funcName)s
    #                         [line:%(lineno)d] %(levelname)s %(message)s'))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    # 设置屏幕打印的格式
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # 设置log保存
    fh = logging.FileHandler(log_file, encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
