# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import onnx
import onnxruntime
import cv2
import torch
import numpy as np
from mmengine import Config, DictAction, mkdir_or_exist
from mmengine.registry import build_from_cfg
from tqdm import tqdm
import struct

from mmpose.registry import DATASETS
from mmpose.utils import register_all_modules
from mmpose.models.utils.siamcc_to_kpt import SimCCToKeypoint

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate data for calibration')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--output-dir',
        default='calib_data',
        type=str,
        help='If there is no display interface, you can save it.')
    parser.add_argument(
        '--nr-sample',
        default=4096,
        type=int,
        help='number of data samples, -1 means get all data in order')
    parser.add_argument(
        '--phase',
        default='val',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--quantize_type',
        default='snpe',
        type=str,
        help='choose data channel type, (n,h,w,c) raw file for snpe, (n,c,h,w) raw file for mnn, png file for npu')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def generate_dup_file_name(out_file):
    """Automatically rename out_file when duplicated file exists.

    This case occurs when there is multiple instances on one image.
    """
    if out_file and osp.exists(out_file):
        img_name, postfix = osp.basename(out_file).rsplit('.', 1)
        exist_files = tuple(
            filter(lambda f: f.startswith(img_name),
                   os.listdir(osp.dirname(out_file))))
        if len(exist_files) > 0:
            img_path = f'{img_name}({len(exist_files)}).{postfix}'
            out_file = osp.join(osp.dirname(out_file), img_path)
    return out_file


def main():
    args = parse_args()
    mkdir_or_exist(args.output_dir)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules()
    
    dataset_list = [
        # # 3d
        '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble/XS__20240517_030629__all__normal__right__1101__0006__undistort_tar__Flora301.json',
        '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble/XS__20230906_030815__pinch__normal__left__1111__0022__undistort_tar__Flora302.json',
        # # # # 2d
        '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/convert2d_to_3d_e2e/hand_train_flora_e2e_20241220_1__1__binocular__lmdb.json',
        '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/convert2d_to_3d_e2e/hand_train_flora_e2e_20250915__1__binocular__lmdb.json',
        
        # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/filter_IK/annotations3d_nimble_fixed/XS__20240401_063058__pinch__normal__left__1110__0006__undistort_tar__Flora301.json',
        # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20240816_073057__pinch__normal__right__1101__0032__undistort_tar__Flora301.json',

        # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20240930_055316__all__normal__right__1101__0007__undistort_tar__Flora301.json',
        # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/convert2d_to_3d_stliu/hand_train_flora_10k_230327_1_cam0_lmdb__point_flora.json',
        # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/convert2d_to_3d_stliu/hand_train_flora_hoi_240716_5k__1__binocular__lmdb.json',
        # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/convert2d_to_3d_e2e/hand_train_flora_e2e_20241220_1__1__binocular__lmdb.json',
    ]
    random.seed(0)
    
    # ipr_module = SimCCToKeypoint(256, 256)    
    rtconfig = onnxruntime.SessionOptions()
    cpu_num_thread = 4
    rtconfig.intra_op_num_threads = cpu_num_thread
    rtconfig.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    providers = ['CPUExecutionProvider']
    onnx_path = "/home/byzhou/code/mmpose/liftnimble_DLT_new2DModel_filterAllData_sameSource_postNorm_WX03_1229_kptAllOnes_filterInvalidHand_WX03_0109_791aa4.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers, sess_options=rtconfig)
    
    count = 0
    for dataset_idx, dataset_path in enumerate(dataset_list):
        cfg['quant_calib_3d_dataset']['data_file_list'] = [dataset_path]    
        seq_length = cfg['quant_calib_3d_dataset']["seq_len"]
        dataset = build_from_cfg(cfg['quant_calib_3d_dataset'], DATASETS)
        if args.nr_sample > 0:
            sample_index = random.sample(range(len(dataset)), args.nr_sample)
        else:
            sample_index = range(len(dataset))
        for sample_idx, sample_id in enumerate(tqdm(sample_index)):
            item = dataset[sample_id]
            # print(item)
            inputs = item['inputs']
            data_samples = item["data_samples"]
            
            for seq_idx in range(seq_length):
                
                count += 1
                img = inputs[seq_idx].unsqueeze(0).numpy().astype(np.float32)
                img_norm = (img - 0.449 * 255) / (0.226 * 255)
                f_scale = np.array([data_samples[seq_idx].meta['virtual_camera'].f[0] / 200]).astype(np.float32)
                f_scale = f_scale[:, np.newaxis, np.newaxis, np.newaxis]
                
                print(f"img_id: {str(data_samples[seq_idx].metainfo['img_id']).zfill(8)}")
                print(f"anno_id: {str(data_samples[seq_idx].id).zfill(8)}")
                
                if seq_idx == 0:
                    mems = np.zeros((1, 384, 1, 1), dtype=np.float32)
                                
                img_id = str(count).zfill(8)

                if args.quantize_type == 'snpe':
                    img = (img - 114.495) / 57.63 
                    img = img.transpose((0, 2, 3, 1))  # (N,H,W,C)
                    img.tofile(os.path.join(args.output_dir, f'{img_id}_input.raw'))
                    f_scale.tofile(os.path.join(args.output_dir, f'{img_id}_f_scale.raw'))
                    mems.tofile(os.path.join(args.output_dir, f'{img_id}_mem_in.raw'))
                elif args.quantize_type == 'mnn':
                    img = (img - 114.495) / 57.63
                    # (N,C,H,W)
                    img.tofile(os.path.join(args.output_dir, f'{img_id}_input.raw'))
                    f_scale.tofile(os.path.join(args.output_dir, f'{img_id}_f_scale.raw'))
                    mems.tofile(os.path.join(args.output_dir, f'{img_id}_mem_in.raw'))
                elif args.quantize_type == 'npu':
                    img = img[0].transpose((1, 2, 0))
                    cur_output_dir = os.path.join(args.output_dir, f'{img_id}')
                    os.makedirs(cur_output_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(cur_output_dir, 'input.png'), img)
                    
                    f_scale_bytes = struct.pack('<f', f_scale[0, 0, 0, 0])
                    binfile = open(os.path.join(cur_output_dir, 'f_scale.bin'), 'wb') #追加写入

                    binfile.write(f_scale_bytes)
                    binfile.close()
                    
                    binfile = open(os.path.join(cur_output_dir, 'mem_in.bin'), 'wb') #追加写入
                    for i in range(384):
                        if i == 0:
                            mems_bytes = struct.pack('<f', mems[0, i, 0, 0])
                        else:
                            mems_bytes += struct.pack('<f', mems[0, i, 0, 0])

                    binfile.write(mems_bytes)
                    binfile.close()
                    
                else:
                    raise ValueError("Unsupported quantize_type")
                
                feat_x, feat_y, rot, svd_pt, mems, score, pred_sigma_reshape = ort_session.run(None, {"input": img_norm, "f_scale": f_scale, "mem_in":mems},) 
            
                
if __name__ == '__main__':
    main()
