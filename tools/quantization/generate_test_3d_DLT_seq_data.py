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
import numpy as np
from mmengine import Config, DictAction, mkdir_or_exist
from mmengine.registry import build_from_cfg
from tqdm import tqdm
import struct

from mmpose.registry import DATASETS
from mmpose.utils import register_all_modules


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
        default='train',
        type=str,
        choices=['train', 'test', 'val'],
        help='phase of dataset to visualize, accept "train" "test" and "val".'
        ' Defaults to "train".')
    parser.add_argument(
        '--quantize_type',
        default='snpe',
        type=str,
        help='choose data channel type, (n,h,w,c) for snpe, (n,c,h,w) for mnn')
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
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    register_all_modules()
    
    dataset_list = [
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/fit_nimble_merge_seqsmooth__binocular_coco/XS__20250924_110720__grab__normal__right__1111__0008__undistort_tar__Flora301_100.json'
            # # flora301
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301.json',
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301.json',
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301.json',
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301.json',
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301.json',
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301.json',
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
            # # # flora302
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora302.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora302.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora302.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora302.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora302.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora302.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora302.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora302.json',
            # # # # flora303
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora303.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora303.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora303.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora303.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora303.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora303.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora303.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora303.json',
            # # # flora304
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora304.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora304.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora304.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora304.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora304.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora304.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora304.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora304.json', 
    ]
    random.seed(0)
    rtconfig = onnxruntime.SessionOptions()
    cpu_num_thread = 4
    rtconfig.intra_op_num_threads = cpu_num_thread
    rtconfig.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    providers = ['CPUExecutionProvider']
    onnx_path = "/home/byzhou/code/mmpose/liftnimble_DLT_new2DModel_filterAllData_sameSource_postNorm_WX03_1229_kptAllOnes_filterInvalidHand_WX03_0109_791aa4.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers, sess_options=rtconfig)
    
    count = 0
    for dataset_idx, dataset_path in enumerate(dataset_list):
        
        cfg['quant_test_3d_dataset']['data_file_list'] = [dataset_path]    
        dataset = build_from_cfg(cfg['quant_test_3d_dataset'], DATASETS)

        print("dataset length ", len(dataset))
        
        if args.nr_sample > 0:
            start_idx = random.randint(0, len(dataset)-args.nr_sample - 1)
            sample_index = range(len(dataset))[start_idx:start_idx+args.nr_sample]
        else:
            sample_index = range(len(dataset))
        
        print("len of sample index ", len(sample_index))
        
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        for sample_idx, sample_id in enumerate(tqdm(sample_index)):

            item = dataset[sample_id]
            # print(item)
            img = item['inputs'].unsqueeze(0).numpy().astype(np.float32)
            data_sample = item["data_samples"]
            
            count += 1
            img_norm = (img - 0.449 * 255) / (0.226 * 255)
            f_scale = np.array([data_sample.meta['virtual_camera'].f[0] / 200]).astype(np.float32)
            f_scale = f_scale[:, np.newaxis, np.newaxis, np.newaxis]

            if sample_idx == 0:
                mems = np.zeros((1, 384, 1, 1), dtype=np.float32)
                        
            img_id = str(data_sample.metainfo['img_id']).zfill(8)
            anno_id = str(data_sample.id).zfill(8)            

            # if img_id != "00000218":
            #     continue
            
            if args.quantize_type == 'snpe':
                mkdir_or_exist(os.path.join(args.output_dir, "NHWC"))
                mkdir_or_exist(os.path.join(args.output_dir, "NCHW"))    
                # for mnn                
                img = (img - 114.495) / 57.63
                img.tofile(os.path.join(args.output_dir, "NCHW", f'{dataset_name}_{img_id}_{anno_id}_input.raw'))
                f_scale.tofile(os.path.join(args.output_dir, "NCHW", f'{dataset_name}_{img_id}_{anno_id}_f_scale.raw'))
                mems.tofile(os.path.join(args.output_dir, "NCHW", f'{dataset_name}_{img_id}_{anno_id}_mem_in.raw'))
                
                # for snpe
                img = img.transpose((0, 2, 3, 1))  # (N,H,W,C)
                mems = mems.transpose((0, 2, 3, 1))
                img.tofile(os.path.join(args.output_dir, "NHWC", f'{dataset_name}_{img_id}_{anno_id}_input.raw'))
                f_scale.tofile(os.path.join(args.output_dir, "NHWC", f'{dataset_name}_{img_id}_{anno_id}_f_scale.raw'))
                mems.tofile(os.path.join(args.output_dir, "NHWC", f'{dataset_name}_{img_id}_{anno_id}_mem_in.raw'))
                mems = mems.transpose((0, 3, 1, 2))

            elif args.quantize_type == 'npu':
                img = img[0].transpose((1, 2, 0))

                cur_output_dir = os.path.join(args.output_dir, f'{dataset_name}_{img_id}_{anno_id}')

                if os.path.exists(cur_output_dir):
                    print("exists already")
                    print(cur_output_dir)
                
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
            
            # update memory
            feat_x, feat_y, rot, svd_pt, mems, score, pred_sigma_reshape = ort_session.run(None, {"input": img_norm, "f_scale": f_scale, "mem_in":mems},) 

                
                

if __name__ == '__main__':
    main()
