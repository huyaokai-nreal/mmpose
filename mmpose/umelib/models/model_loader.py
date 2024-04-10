# flake8: noqa
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from mmpose.umelib.data_utils import fs
from . import feature_extractor as fe
from . import regressor as reg
from . import skeleton_encoder as se
from . import temporal as tem
from . import texture_to_coord as t2c
from .model_opts import ModelOpts
from .umetrack_model import UmeTrackModel, UmeTrackModel_coord

# from .umetrack_model_origin import (UmeTrackModel_CCF, UmeTrackModel_coord,
#                                     UmeTrackModel_CrossView,
#                                     UmeTrackModel_Fuse, UmeTrackModel_TM,
#                                     UmeTrackModel_TM_New, UmeTrackModel_TM_Res,
#                                     UmeTrackModel_TM_Res_New,
#                                     UmeTrackModel_TM_Res_New_Fine_Tune,
#                                     UmeTrackModel_TM_Res_New_Fine_Tune_wT,
#                                     UmeTrackModel_TM_Res_sv,
#                                     UmeTrackModel_TM_sv)


def _get_n_input_channels(model_opts: ModelOpts, use_skel: bool) -> int:
    n = model_opts.nImageFeatureChannels
    if use_skel:
        n = n + model_opts.nSkeletonFeatureChannels

    return n


def _create_regressor(
    model_opts: ModelOpts,
    feature_sizes: Tuple[int, int],
    use_skel: bool,
    predict_skel_scale: bool,
):
    if use_skel:
        assert model_opts.nSkeletonFeatureChannels != 0

    n_in = _get_n_input_channels(model_opts, use_skel=use_skel)
    reg_out_indices, n_out = reg.get_output_index_ranges(
        model_opts, predict_skel_scale=predict_skel_scale)
    return reg.PoseRegressor(
        n_channels_in=n_in,
        n_output_dims=n_out,
        output_index_ranges=reg_out_indices,
        n_blocks=model_opts.nPoseRegressionBlocks,
        n_wrist_rigid_pts=model_opts.nWristRigidPts,
        feature_map_sizes=feature_sizes,
    )


def _create_regressor_deconv(
    model_opts: ModelOpts,
    feature_sizes: Tuple[int, int],
    use_skel: bool,
    predict_skel_scale: bool,
):
    if use_skel:
        assert model_opts.nSkeletonFeatureChannels != 0

    n_in = _get_n_input_channels(model_opts, use_skel=use_skel)
    reg_out_indices, n_out = reg.get_output_index_ranges(
        model_opts, predict_skel_scale=predict_skel_scale)
    return reg.PoseRegressor_Deconv(
        n_channels_in=n_in,
        n_output_dims=n_out,
        output_index_ranges=reg_out_indices,
        n_blocks=model_opts.nPoseRegressionBlocks,
        n_wrist_rigid_pts=model_opts.nWristRigidPts,
        feature_map_sizes=feature_sizes,
    )


def load_pretrained_model_tm_sv(model_path=None):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    temporal_sv = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_k_sv = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM_sv(
        feature_extractor=feature_extractor,
        temporal=temporal,
        temporal_sv=temporal_sv,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_k_sv=regressor_k_sv,
        regressor_u=regressor_u,
    )

    with fs.open(model_path, 'rb') as fp:

        model_state_dict = torch.load(fp, map_location=torch.device('cpu'))
        if 'state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['state_dict']
    umetrack_model.load_state_dict(model_state_dict)

    return umetrack_model


def load_pretrained_model_tm(model_path=None):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM(
        feature_extractor=feature_extractor,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_u=regressor_u,
    )

    with fs.open(model_path, 'rb') as fp:

        model_state_dict = torch.load(fp, map_location=torch.device('cpu'))
        if 'state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['state_dict']
    umetrack_model.load_state_dict(model_state_dict)

    return umetrack_model


def load_pretrained_model_tm_res(model_path=None):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM_Res((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM_Res(
        feature_extractor=feature_extractor,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_u=regressor_u,
    )

    with fs.open(model_path, 'rb') as fp:

        model_state_dict = torch.load(fp, map_location=torch.device('cpu'))
        if 'state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['state_dict']

    umetrack_model.load_state_dict(model_state_dict)

    return umetrack_model


def load_pretrained_model_tm_res_sv(model_path=None):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM_Res((96, 96),
                                                   ModelOpts(),
                                                   use_bn=False)
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_k_sv = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM_Res_sv(
        feature_extractor=feature_extractor,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_k_sv=regressor_k_sv,
        regressor_u=regressor_u,
    )

    with fs.open(model_path, 'rb') as fp:

        model_state_dict = torch.load(fp, map_location=torch.device('cpu'))
        if 'state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['state_dict']

    umetrack_model.load_state_dict(model_state_dict)

    return umetrack_model


def load_pretrained_model_tm_res_new_fine_tune(model_path=None):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM_Res_New((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_k_sv = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )
    umetrack_model = UmeTrackModel_TM_Res_New_Fine_Tune(
        feature_extractor=feature_extractor,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_k_sv=regressor_k_sv,
        regressor_u=regressor_u,
    )
    with fs.open(model_path, 'rb') as fp:
        model_state_dict = torch.load(fp, map_location=torch.device('cpu'))
        if 'state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['state_dict']
    umetrack_model.load_state_dict(model_state_dict)
    return umetrack_model


def load_pretrained_model_tm_res_new_fine_tune_wT(model_path=None):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM_Res_New((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    temporal_sv = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_k_sv = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )
    umetrack_model = UmeTrackModel_TM_Res_New_Fine_Tune_wT(
        feature_extractor=feature_extractor,
        temporal=temporal,
        temporal_sv=temporal_sv,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_k_sv=regressor_k_sv,
        regressor_u=regressor_u,
    )
    with fs.open(model_path, 'rb') as fp:
        model_state_dict = torch.load(fp, map_location=torch.device('cpu'))
        if 'state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['state_dict']
    umetrack_model.load_state_dict(model_state_dict)
    return umetrack_model


# 测试
def load_pretrained_model(model_path=None):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel(
        feature_extractor=feature_extractor,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_u=regressor_u,
    )
    # import ipdb;ipdb.set_trace()
    with fs.open(model_path, 'rb') as fp:

        model_state_dict = torch.load(fp, map_location=torch.device('cpu'))
        if 'state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['state_dict']
    umetrack_model.load_state_dict(model_state_dict)

    return umetrack_model


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.001)
        # nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
        nn.init.constant_(m.bias, 0)


def create_model():
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel(
        feature_extractor=feature_extractor,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_u=regressor_u,
    )
    umetrack_model.apply(init_weights)

    return umetrack_model


# 训练 #
def create_model_coord():
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor((96, 96), ModelOpts())
    texture_to_coord = t2c.create_t2c()
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )

    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_coord(
        feature_extractor=feature_extractor,
        texture_to_coord=texture_to_coord,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_u=regressor_u,
    )
    umetrack_model.apply(init_weights)

    return umetrack_model


def create_model_tm_res():
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM_Res((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM_Res(
        feature_extractor=feature_extractor,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_u=regressor_u,
    )
    umetrack_model.apply(init_weights)

    return umetrack_model


def create_model_tm_res_sv(use_bn=True):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM_Res((96, 96),
                                                   ModelOpts(),
                                                   use_bn=use_bn)
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_k_sv = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM_Res_sv(
        feature_extractor=feature_extractor,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_k_sv=regressor_k_sv,
        regressor_u=regressor_u,
    )
    umetrack_model.apply(init_weights)

    return umetrack_model


def create_model_tm_res_new():
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM_Res_New((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM_Res_New(
        feature_extractor=feature_extractor,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_u=regressor_u,
    )
    umetrack_model.apply(init_weights)

    return umetrack_model


def create_model_tm_res_new_fine_tune(model_path):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM_Res_New((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_k_sv = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM_Res_New_Fine_Tune(
        feature_extractor=feature_extractor,
        temporal=temporal,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_k_sv=regressor_k_sv,
        regressor_u=regressor_u,
    )
    model_dict = umetrack_model.state_dict()
    # optim_pram = []
    with fs.open(model_path, 'rb') as fp:
        model_state_dict = torch.load(fp, map_location=torch.device('cpu'))
        if 'state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['state_dict']
        state_dict = {
            k: v
            for k, v in model_state_dict.items() if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        umetrack_model.load_state_dict(model_dict)
        for k, v in umetrack_model.named_parameters():
            if k in model_state_dict.keys():
                # print(k,True)
                v.requires_grad = False  #固定参数
            # else:
            #     optim_pram.append(v)
        # assert 1==2
    return umetrack_model


def create_model_tm_res_new_fine_tune_wT(model_path):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM_Res_New((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    temporal_sv = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_k_sv = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM_Res_New_Fine_Tune_wT(
        feature_extractor=feature_extractor,
        temporal=temporal,
        temporal_sv=temporal_sv,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_k_sv=regressor_k_sv,
        regressor_u=regressor_u,
    )
    model_dict = umetrack_model.state_dict()
    # optim_pram = []
    with fs.open(model_path, 'rb') as fp:
        model_state_dict = torch.load(fp, map_location=torch.device('cpu'))
        if 'state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['state_dict']
        state_dict = {
            k: v
            for k, v in model_state_dict.items() if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        umetrack_model.load_state_dict(model_dict)
        for k, v in umetrack_model.named_parameters():
            if k in model_state_dict.keys():
                # print(k,True)
                v.requires_grad = False  #固定参数
            # else:
            #     print(k)
        # assert 1==2
    return umetrack_model


def create_model_tm_res_new_fine_tune_wT_deconv(model_path):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM_Res_New((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    temporal_sv = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_k_sv = _create_regressor_deconv(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )

    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM_Res_New_Fine_Tune_wT(
        feature_extractor=feature_extractor,
        temporal=temporal,
        temporal_sv=temporal_sv,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_k_sv=regressor_k_sv,
        regressor_u=regressor_u,
    )
    model_dict = umetrack_model.state_dict()
    # optim_pram = []
    with fs.open(model_path, 'rb') as fp:
        model_state_dict = torch.load(fp, map_location=torch.device('cpu'))
        if 'state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['state_dict']
        state_dict = {
            k: v
            for k, v in model_state_dict.items() if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        umetrack_model.load_state_dict(model_dict)
        for k, v in umetrack_model.named_parameters():
            if k in model_state_dict.keys():
                # print(k,True)
                v.requires_grad = False  #固定参数
            # else:
            #     print(k)
        # assert 1==2
    return umetrack_model


def create_model_tm_res_new_fine_tune_wT_sub(model_path):
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM_Res_Sub((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    temporal_sv = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_k_sv = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM_Res_New_Fine_Tune_wT(
        feature_extractor=feature_extractor,
        temporal=temporal,
        temporal_sv=temporal_sv,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_k_sv=regressor_k_sv,
        regressor_u=regressor_u,
    )
    model_dict = umetrack_model.state_dict()
    # optim_pram = []
    with fs.open(model_path, 'rb') as fp:
        model_state_dict = torch.load(fp, map_location=torch.device('cpu'))
        if 'state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['state_dict']
        state_dict = {
            k: v
            for k, v in model_state_dict.items() if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        umetrack_model.load_state_dict(model_dict)
        for k, v in umetrack_model.named_parameters():
            if k in model_state_dict.keys():
                # print(k,True)
                v.requires_grad = False  #固定参数
            # else:
            #     print(k)
        # assert 1==2
    return umetrack_model


def create_model_tm_sv():
    model_opts = ModelOpts()
    feature_extractor = fe.FeatureExtractor_TM((96, 96), ModelOpts())
    temporal = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    temporal_sv = tem.create_temporal_model(
        model_opts,
        feature_extractor.output_feature_sizes,
    )
    skeleton_encoder = se.SkeletonEncoder([
        model_opts.nSkeletonFeatureChannels,
        *feature_extractor.output_feature_sizes
    ], )
    regressor_k = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_k_sv = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=True,
        predict_skel_scale=False,
    )
    regressor_u = _create_regressor(
        model_opts,
        feature_extractor.output_feature_sizes,
        use_skel=False,
        predict_skel_scale=True,
    )

    umetrack_model = UmeTrackModel_TM_sv(
        feature_extractor=feature_extractor,
        temporal=temporal,
        temporal_sv=temporal_sv,
        skeleton_encoder=skeleton_encoder,
        regressor_k=regressor_k,
        regressor_k_sv=regressor_k_sv,
        regressor_u=regressor_u,
    )
    umetrack_model.apply(init_weights)

    return umetrack_model
