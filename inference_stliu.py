import onnxruntime
import numpy as np
from inference_nimble_python import forward_3d, get_rel_kpt3d, rot9D_to_matirx, decode_svd

rtm2d_model_path = "/data/stliu/mmpose_simliar_wx10/liftnimble_pcl_mano_one_dbf335.onnx"
lift_model_path = "/data/stliu/mmpose_simliar_wx10/liftnimble_pcl_mano_two_ddeaaa.onnx"

device_id = 0
providers = [
    ('CUDAExecutionProvider', {
        'device_id': device_id,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'DEFAULT',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

rtm2d_model = onnxruntime.InferenceSession(
    rtm2d_model_path, providers=providers)
liftnet_model = onnxruntime.InferenceSession(
    lift_model_path, providers=providers)


image_path = "/data/stliu/mmpose_simliar_wx10/info_image.npy"
intrix_matrix_path = "/data/stliu/mmpose_simliar_wx10/intrix_matrix.npy"
f_scale_path = "/data/stliu/mmpose_simliar_wx10/f_scale.npy"
left_R_path = "/data/stliu/mmpose_simliar_wx10/left_R.npy"


mems = np.zeros((1, 384, 1, 1)).astype(np.float32)
shape = np.zeros((1,1)).astype(np.float32)

image_info = np.load(image_path)
intrix_matrix = np.load(intrix_matrix_path)
f_scale = np.load(f_scale_path)
left_R = np.load(left_R_path)
category_name = "left_hand"

def cal_kpt_weight(sigma):
    weight_num = 42
    kpt_weight = np.eye(weight_num)[np.newaxis, ...]
    sigma_kpt = np.mean(sigma, axis=-1)

    exp_sigma_kpt = np.exp(sigma_kpt - np.max(sigma_kpt, axis=1, keepdims=True))
    sigma_kpt_softmax = exp_sigma_kpt / np.sum(exp_sigma_kpt, axis=1, keepdims=True)
    sigma_kpt_softmax = np.repeat(sigma_kpt_softmax[:, :, np.newaxis], 2, axis=2).reshape(sigma_kpt.shape[0], -1)

    indices = np.arange(weight_num)
    kpt_weight[:, indices, indices] = sigma_kpt_softmax * 21
    return kpt_weight

def simcc_to_keypoint(x, y, feat_w=256, feat_h=256):
    # 创建 linspace_x 和 linspace_y
    linspace_x = np.arange(0.0, 1.0 * feat_w, 1.0) / feat_w
    linspace_y = np.arange(0.0, 1.0 * feat_h, 1.0) / feat_h

    def softmax(featmaps, axis=2):
        exp_fm = np.exp(featmaps - np.max(featmaps, axis=axis, keepdims=True))
        return exp_fm / np.sum(exp_fm, axis=axis, keepdims=True)

    x = softmax(x, axis=2)
    y = softmax(y, axis=2)
    pred_x = np.sum(x * linspace_x, axis=-1, keepdims=True)
    pred_y = np.sum(y * linspace_y, axis=-1, keepdims=True)
    return pred_x * 128, pred_y * 128

def get_3d_kpt(hand3d_rel, cood_2d, intrix_matrix, W):
    batch_size, K = hand3d_rel.shape[0], hand3d_rel.shape[1]
    cood_2d = np.concatenate((cood_2d, np.ones((batch_size, K, 1))), axis=-1)
    uv_cood_leftmatrix = np.matmul(np.linalg.inv(intrix_matrix), cood_2d.transpose(0, 2, 1)).transpose(0, 2, 1)[..., :2]

    A = np.zeros((batch_size, 2 * K, 3))
    A[:, ::2, 0] = -1
    A[:, 1::2, 1] = -1
    A[:, ::2, 2] = uv_cood_leftmatrix[:, :, 0].reshape(batch_size, K)
    A[:, 1::2, 2] = uv_cood_leftmatrix[:, :, 1].reshape(batch_size, K)

    B = np.zeros((batch_size, 2 * K, 1))
    B[:, ::2, 0] = hand3d_rel[:, :, 0] - hand3d_rel[:, :, 2] * uv_cood_leftmatrix[:, :, 0]
    B[:, 1::2, 0] = hand3d_rel[:, :, 1] - hand3d_rel[:, :, 2] * uv_cood_leftmatrix[:, :, 1]

    part_1 = np.linalg.inv(np.matmul(np.matmul(A.transpose(0, 2, 1), W), A))
    part_2 = np.matmul(np.matmul(A.transpose(0, 2, 1), W), B)
    result = np.matmul(part_1, part_2).transpose(0, 2, 1)
    hand3d = hand3d_rel + result
    return hand3d




for i in range(image_info.shape[0]):
    image = image_info[i:i+1,...]
    
    inputs_rtm2d = {
        rtm2d_model.get_inputs()[0].name: image,
    }
    feat_x, feat_y, feat = rtm2d_model.run(None, inputs_rtm2d)

    inputs_liftnet = {
        liftnet_model.get_inputs()[0].name: feat,
        liftnet_model.get_inputs()[1].name: f_scale,
        liftnet_model.get_inputs()[2].name: mems,
    }
    
    rot, svd_pt, mems, score, sigma = liftnet_model.run(None, inputs_liftnet)
    local_matrix = rot9D_to_matirx(rot)
    root_matrix = decode_svd(svd_pt)[:,:3,:3].numpy()
    kpt_weight = cal_kpt_weight(sigma)
    
    kpt = forward_3d(shape, local_matrix)
    kpt_rel = get_rel_kpt3d(kpt).numpy()

    pred_x, pred_y = simcc_to_keypoint(feat_x, feat_y)
    add_matrix = np.eye(3)[None,:]
    if "left" in category_name:
        pred_x = 127 - pred_x
        add_matrix[:, 0,0] = -1
    
    root_matrix = np.matmul(add_matrix, root_matrix)
    kpt_rel = np.matmul(root_matrix, kpt_rel.transpose(0, 2, 1)).transpose(0, 2, 1)
    
    cood_2d = np.concatenate((pred_x, pred_y), axis=-1)
    kpt3d_virtual = get_3d_kpt(kpt_rel, cood_2d, intrix_matrix, kpt_weight)
    
    left_R_inv = np.linalg.inv(left_R)
    kpt3d_world = np.matmul(left_R_inv, kpt3d_virtual.transpose(0, 2, 1)).transpose(0, 2, 1)
    
    from IPython import embed; embed()


