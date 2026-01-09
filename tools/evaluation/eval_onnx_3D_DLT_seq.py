import argparse
import os
import os.path as osp
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import onnx
import onnxruntime
import torch
import cv2
from itertools import zip_longest
import copy
from tqdm import tqdm

import mmcv
import mmengine
import numpy as np
from mmengine import Config, DictAction, mkdir_or_exist
from mmengine.registry import build_from_cfg, init_default_scope
from mmengine.structures import InstanceData
from nreal_data_tool import LmdbClient
from nreal_data_tool.utils.camera import NoDistortion, PinholePlaneCameraModel
from mmpose.models.utils.pose_solver import (get_kpt_depth,
                                             get_kpt_depth_binocular,
                                             get_kpt_depth_binocular63,
                                             get_root_depth)
from mmpose.structures.bbox import bbox_cs2xyxy

from typing import Optional
from mmengine.evaluator import Evaluator
from mmpose.registry import DATASETS, VISUALIZERS
from mmpose.structures import PoseDataSample
from mmpose.models.utils.siamcc_to_kpt import SimCCToKeypoint3D, SimCCToKeypoint
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptMultiConfig, PixelDataList, SampleList)
from mmpose.utils import register_all_modules
from mmpose.models.heads.nimble.nimble_utils import (SkeletonEncoder,
                                                     _gen_rigid_features,
                                                     batch_rodrigues,
                                                     convert_vector2matrix,
                                                     decode_svd,
                                                     euler_angles_to_matrix,
                                                     rot6D_to_matirx,
                                                     rot9D_to_matirx)
from mmpose.models.heads.nimble.simple_NIMBLELayer import sim_NIMBLELayer

import multiprocessing

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate data for calibration')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def add_pred_to_datasample_nimble(
        self, pre_info,
        batch_pred_fields: Optional[PixelDataList],
        batch_data_samples: SampleList) -> SampleList:
    
    pred, pred_bino_kp2d, sigmas = pre_info
    batch_pred_instances = []
    for b in range(pred.shape[0]):
        keypoints = pred_bino_kp2d[b:b + 1, ...]  # gt为左目信息
        batch_pred_instances.append(
            InstanceData(
                keypoints3d=pred[b:b + 1, ...],
                keypoints3d_scores=sigmas[b:b + 1, ...],
                keypoints=keypoints,
                keypoint_scores=torch.ones((1, 21)),
            ))

    assert len(batch_pred_instances) == len(batch_data_samples)
    if batch_pred_fields is None:
        batch_pred_fields = []

    for pred_instances, pred_fields, data_sample in zip_longest(
            batch_pred_instances, batch_pred_fields, batch_data_samples):
        pred_instances.keypoints3d = pred_instances.keypoints3d.cpu(
        ).numpy()
        pred_instances.keypoints3d_scores = \
            pred_instances.keypoints3d_scores.cpu().numpy()
        pred_instances.keypoints = pred_instances.keypoints.cpu().numpy()
        pred_instances.keypoints = np.concatenate(
            (pred_instances.keypoints, pred_instances.keypoints3d[...,
                                                                    2:]),
            axis=-1)
        ori_cam = data_sample.meta['ori_camera']
        hand2d_gt = ori_cam.eye_to_window(data_sample.gt_instances.keypoints3d[0])

        # if 'XS__20230830_070648__all__normal__right__1111__0005__00000007__cam1__2220' in data_sample.img_path:
        #     print("from add_pred_to_datasample_nimble ", pred_instances.keypoints)
        #     print("hand2d_gt ", hand2d_gt)
        #     print("initial input ", pred_bino_kp2d)       

        
        # if 'virtual_camera' in data_sample.meta:
        #     ori_cam = data_sample.meta['ori_camera']
        #     virtual_cam = data_sample.meta['virtual_camera']
        #     # vritual_point_world = virtual_cam.world_to_eye(hand3d_gt_sin.cpu().numpy())
        #     # vritual_point = virtual_cam.eye_to_window(vritual_point_world)
        #     kpt_norm_eye = virtual_cam.window_to_eye(
        #         pred_instances.keypoints[0,:, :2])
        #     kpt_norm_world = virtual_cam.eye_to_world(kpt_norm_eye)
        #     kpt2d_ori = ori_cam.eye_to_window(kpt_norm_world)
        #     pred_instances.keypoints[0][..., :2] = kpt2d_ori
        
        # data_sample.gt_instances.keypoints = np.concatenate(
        #     (hand2d_gt[np.newaxis, :], data_sample.gt_instances.keypoints3d[..., 2:]),
        #     axis=-1)
        pred_instances.keypoint_scores = np.ones(
            (1, pred_instances.keypoints.shape[1]))
        # if data_sample.meta['flipped']:
        #     pred_kpt = pred_instances.keypoints[0]
        #     gt_kpt = data_sample.gt_instances.keypoints[0]
        #     pred_kpt[..., 0] = (
        #         data_sample.meta['frame_width'] - 1 - pred_kpt[..., 0])
        #     gt_kpt[..., 0] = (
        #         data_sample.meta['frame_width'] - 1 - gt_kpt[..., 0])
        data_sample.pred_instances = pred_instances
    return batch_data_samples

nimble_layer = sim_NIMBLELayer(
    device='cuda:0',
    shape_ncomp=1,
    pose_ncomp=19*9,
    use_pose_pca=True,
    reg_shape_type=1)

kp_index = [
    0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22,
    23, 24
]

scale_parameter = 1000

rigid_samples = _gen_rigid_features()

def postprocess(rot_vector_t, svd_points, hand_type,
            left_hand,
            left_R,
            f_scale,
            uv_coord,
            intrix_matrix,
            kpt3d_weight,
            only_pre=True):

    B = 1
    matrix_svd = decode_svd(
        svd_points,
        rigid_samples,
    )

    cuda_device = torch.device("cuda:0")

    pre_root_xyz = matrix_svd[:, 0:3, 3].to(cuda_device)
    pre_root_matrix = matrix_svd[:, 0:3, 0:3].to(cuda_device)

    pre_local_matrix = rot9D_to_matirx(rot_vector_t).to(cuda_device)
    
    shape_vector = torch.zeros((B, 1)).to(cuda_device)

    mask = left_hand == 1
    add_matrix = torch.eye(3).unsqueeze(0).expand(1, -1,
                                                -1).to(cuda_device)
    add_matrix[mask, 0, 0] = -add_matrix[mask, 0, 0]

    pre_root_xyz = torch.matmul(add_matrix,
                                pre_root_xyz.unsqueeze(-1))[:, :, 0]
    pre_root_matrix = torch.matmul(add_matrix, pre_root_matrix)

    pre_root_zeros = torch.zeros((1, 3)).to(cuda_device)
    left_R_zeros = torch.eye(3)[None, :, :].repeat(1, 1, 1)
    
    uv_coord[mask, :, 0] = (127 - uv_coord[mask, :, 0])
    def get_nimble_3d(root_xyz, root_matrix, local_matrix, shape_vector,
                        left_R, f_scale):

        nimble_layer.to(cuda_device)
        _, bone_joints = nimble_layer.forward_simple(
            local_matrix.to(cuda_device), shape_vector.to(cuda_device))  # 通过局部点旋转，scale，将默认局部手型得到实际局部手型
        
        rebuild_joints = bone_joints[:, kp_index, :].cpu()
        root_rebuild_joints = rebuild_joints[:, 0:1, :]
        rebuild_joints_temp = rebuild_joints - root_rebuild_joints
        
        root_matrix = torch.matmul(torch.inverse(left_R), root_matrix.cpu())
        rebuild_joints_temp = torch.matmul(rebuild_joints_temp,
                                            root_matrix.transpose(1, 2))
        rebuild_joints_with_scale = rebuild_joints_temp / scale_parameter
        
        new_root_xyz = torch.bmm(
            root_xyz.unsqueeze(1).cpu(),
            torch.inverse(left_R).permute(0, 2, 1))
        new_root_xyz = new_root_xyz.mul(f_scale[:, None,
                                                    None].repeat(1, 1, 3))
        
        xyz_point = rebuild_joints_with_scale + new_root_xyz
                
        return xyz_point
    
    def get_root_xyz(hand3d_rel, cood_2d, intrix_matrix, W):

        batch_size, K = hand3d_rel.shape[0], hand3d_rel.shape[1]
        cuda_device = cood_2d.device
        cood_2d = torch.concat(
            (cood_2d, torch.ones(batch_size, K, 1).to(cuda_device)),
            dim=-1)
        uv_cood_leftmatrix = torch.matmul(
            torch.inverse(intrix_matrix),
            cood_2d.permute(0, 2, 1)).permute(0, 2,
                                                1)[..., :2].to(cuda_device)

        A = torch.zeros((batch_size, 2 * K, 3), device=cuda_device)
        A[:, ::2, 0] = -1
        A[:, 1::2, 1] = -1
        A[:, ::2, 2] = uv_cood_leftmatrix[:, :, 0].view(batch_size, K)
        A[:, 1::2, 2] = uv_cood_leftmatrix[:, :, 1].view(batch_size, K)

        B = torch.zeros((batch_size, 2 * K, 1), device=cuda_device)
        B[:, ::2,
            0] = hand3d_rel[:, :,
                            0] - hand3d_rel[:, :,
                                            2] * uv_cood_leftmatrix[:, :, 0]
        B[:, 1::2,
            0] = hand3d_rel[:, :,
                            1] - hand3d_rel[:, :,
                                            2] * uv_cood_leftmatrix[:, :, 1]
        # result = torch.matmul(torch.matmul(torch.inverse(torch.matmul(A.permute(0,2,1), A)), A.permute(0,2,1)), B).permute(0,2,1)

        part_1 = torch.inverse(
            torch.matmul(
                torch.matmul(torch.matmul(A.permute(0, 2, 1), W), W), A))
        part_2 = torch.matmul(
            torch.matmul(torch.matmul(A.permute(0, 2, 1), W), W), B)
        result = torch.matmul(part_1, part_2).permute(0, 2, 1)

        hand3d = hand3d_rel + result
        return hand3d
    
    if only_pre:
        hand3d_wo_root = get_nimble_3d(pre_root_zeros, pre_root_matrix,
                        pre_local_matrix, shape_vector,
                        left_R_zeros, f_scale[:, 0, 0, 0])
        pred_3d_in_virtual = get_root_xyz(hand3d_wo_root, uv_coord,
                                        intrix_matrix, kpt3d_weight)
                
        pre_all__xyz = torch.matmul(
            torch.inverse(left_R),
            pred_3d_in_virtual.float().permute(0, 2, 1)).permute(0, 2, 1)
        
        return pre_all__xyz

import random


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    register_all_modules()
    cfg[f'val_dataloader'].dataset.pipeline[
        -1].pack_transformed = True
    
    dataset_list = [
            '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_070648__all__normal__right__1111__0005__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_071804__all__bright__left__1111__0005__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072334__pinch__dark__right__1111__0005__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_072715__pinch__normal__left__1111__0005__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073026__pinch__bright__right__1111__0005__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073556__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_073857__pinch__normal__right__1111__0005__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_074601__pinch__bright__left__1111__0005__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_075055__all__bright__right__1111__0019__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_075728__all__dark__left__1111__0019__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_080158__pinch__normal__right__1111__0019__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_080536__pinch__bright__left__1111__0019__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_080910__pinch__normal__left__1111__0019__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_081151__pinch__dark__right__1111__0019__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_081427__pinch__normal__right__1111__0019__undistort_tar__Flora301.json',
            # '/data/AI_DATA_WX/data_hand/hand_keypoint/annotations3d/Flora_bmk_fix/XS__20230830_081909__pinch__bright__left__1111__0019__undistort_tar__Flora301.json'
    ]
    random.seed(0)

    onnx_path = "/home/byzhou/code/mmpose/liftnimble_DLT_new2DModel_filterAllData_sameSource_postNorm_WX03_1229_kptAllOnes_WXA100_0105_9ae49f.onnx"
    cpu_num_thread = 4
    rtconfig = onnxruntime.SessionOptions()
    rtconfig.intra_op_num_threads = cpu_num_thread
    rtconfig.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    providers = ['CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=providers, sess_options=rtconfig)
    
    input_size = cfg['codec']['input_size'][0]
    ipr_module = SimCCToKeypoint(input_size*2, input_size*2)
    
    evaluator = Evaluator([       
        dict(
           type='MPJPEV2',
           mode='mpjpe',
           scale_metric=False,
           with_tag=False,
           rearrange_result=True,
       ),
       dict(type='MPJPEV2', mode='p-mpjpe', prefix='1')])
    valid_num = 0
    pbar = tqdm(total=len(dataset_list))
    pbar.set_description("Eval Onnx: ")
    
    for dataset_path in dataset_list:
        
        cfg['val_3d_dataset']['data_file_list'] = [dataset_path]    
        dataset = build_from_cfg(cfg['val_3d_dataset'], DATASETS)
        
        print("dataset length ", len(dataset))
        
        sample_index = range(len(dataset))
                
        for sample_idx, sample_id in enumerate(tqdm(sample_index)):
            
            item = dataset[sample_id]
            img = item['inputs'].unsqueeze(0).numpy().astype(np.float32)
            data_sample = item["data_samples"]
            
            img_norm = (img - 0.449 * 255) / (0.226 * 255)
            f_scale = np.array([data_sample.meta['virtual_camera'].f[0] / 200]).astype(np.float32)
            f_scale = f_scale[:, np.newaxis, np.newaxis, np.newaxis]

            if sample_idx == 0:
                mems = np.zeros((1, 384, 1, 1), dtype=np.float32)
            
            # update memory
            feat_x, feat_y, angle, svd, mems, score, pred_sigma_reshape = ort_session.run(None, {"input": img_norm, "f_scale": f_scale, "mem_in":mems},) 

            pred_x, pred_y = ipr_module(torch.from_numpy(feat_x), torch.from_numpy(feat_y))
            
            pred_x = pred_x.numpy()
            pred_y = pred_y.numpy()
            uv_coord = np.concatenate([pred_x[0], pred_y[0]], axis=1)
            uv_coord = uv_coord * np.array([128, 128])

            uv_coord = uv_coord[None, :, :]
            # for 3d postprocess
        
            angle = angle.reshape(1, 19, 9)
            svd = svd.reshape(1, 21)
            score = score.reshape(1, 1)
            sigma = pred_sigma_reshape.reshape(1, 21, 3)
            
            sigma = torch.from_numpy(sigma)
            weight_num = sigma.shape[1] * 2
            kpt_weight = torch.eye(weight_num).unsqueeze(0).repeat(1, 1, 1)
            
            sigma_kpt = torch.mean(sigma, dim=-1)
            sigma_kpt_softmax = torch.softmax(sigma_kpt, dim=1)
            sigma_kpt_softmax = sigma_kpt_softmax.unsqueeze(2).repeat(
                1, 1, 2).view(sigma_kpt.shape[0], -1)
            indices = torch.arange(weight_num)
            kpt_weight[:, indices, indices] = sigma_kpt_softmax * 21
            hand_type = "left_hand" if data_sample.meta['category_id'] == 1 else "right_hand"
            left_hand = [1 if data_sample.meta['category_id'] == 1 else 0]
            
            vritual_camera = data_sample.meta['virtual_camera']
            intrix_m = torch.from_numpy(np.array([[vritual_camera.f[0], 0, vritual_camera.c[0]],
                                    [0, vritual_camera.f[1], vritual_camera.c[1]],
                                    [0, 0, 1]]))[None, :].float()

            f_scale = torch.from_numpy(np.array([vritual_camera.f[0] / 200])).float()[:,None,None,None]
            left_R = np.linalg.inv(vritual_camera.camera_to_world_xf[:3, :3])

            hand3d_pred = postprocess(torch.from_numpy(angle), torch.from_numpy(svd), 
                        hand_type, torch.from_numpy(np.array(left_hand)), torch.from_numpy(left_R).float().unsqueeze(0), f_scale, torch.from_numpy(uv_coord).float(), intrix_m, kpt_weight)
                        
            preds = [
                InstanceData(
                    keypoints3d=hand3d_pred.numpy(),
                    keypoints3d_scores=np.ones((1, 21)),
                    keypoints=data_sample.gt_instances.keypoints,
                    keypoint_scores=np.ones((1, 21)),
                )]
            # print(data_sample)
            data_sample.pred_instances = preds[0]
            evaluator.process(data_samples=[data_sample], data_batch=None)
            valid_num += 1

            #     break
            # break

    metrics = evaluator.evaluate(valid_num)
    print(metrics)

if __name__ == '__main__':
    main()
