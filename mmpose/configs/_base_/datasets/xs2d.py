# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa
datasets_info = {
    'train_data': {
        'public': [
            'data_hand/hand_keypoint/annotations/train_hanco_rgb_gesture_lmdb_refresh.json',  #84k
            'data_hand/hand_keypoint/annotations/train_nreal_baidu1_gesture_right_0930_lmdb.json',  #13.4k
            'data_hand/hand_keypoint/annotations/train_nreal_baidu2_gesture_right_1014_lmdb.json',  #12k
        ],
        'ella': [
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_baidu_0107_2_1_lmdb.json',  #16.8k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_baidu_220216_2_2_lmdb.json',  #32k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_baidu_220216_2_3_lmdb.json',  #12k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_1111_1_1_twohand_lmdb.json',  #13.2k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_1118_1_2_twohand_lmdb.json',  #24k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_1125_1_3_twohand_lmdb.json',  #29.7k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_1202_1_4_twohand_lmdb.json',  #31.8k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_1209_1_5_twohand_lmdb.json',  #24.3k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_1216_1_6_twohand_lmdb.json',  #25.1k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_1223_1_7_twohand_lmdb.json',  #27.8k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_1230_1_8_twohand_lmdb.json',  #17.2k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_0113_1_9_twohand_lmdb.json',  #24k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_0127_1_10_twohand_lmdb.json',  #25k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_0218_1_11_twohand_lmdb.json',  #16k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_0304_1_12_twohand_lmdb.json',  #22k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_0905_1_13~15_bad_data_twohand_lmdb.json',  #26.5k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_0906_1_16~20_bad_data_twohand_lmdb.json',  #88k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_0906_1_21~22_bad_data_twohand_lmdb.json',  #70k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_0905_1_23~29_bad_data_twohand_lmdb.json',  #116k
            'data_hand/hand_keypoint/annotations/train_nreal_gesture_0916_1_30_bad_case_twohand_lmdb.json',
        ],
        'flora': [
            'data_hand/hand_keypoint/annotations/hand_train_flora_10k_230327_1_cam0_lmdb__point_flora.json',  #10k
            'data_hand/hand_keypoint/annotations/hand_train_flora_20k_230822_1_cam0_lmdb.json',
            'data_hand/hand_keypoint/annotations/hand_train_flora_20k_230829_1_cam0_lmdb.json',
            'data_hand/hand_keypoint/annotations/hand_train_flora_20k__230914__1__cam0__lmdb.json',
            'data_hand/hand_keypoint/annotations/hand_train_flora_keypoint_231027_20k__1__binocular__lmdb.json',
            'data_hand/hand_keypoint/annotations/hand_train_flora_keypoint_decoration_1_231017_20k__1__binocular__lmdb.json',
            'data_hand/hand_keypoint/annotations/hand_train_flora_keypoint_bottom_1_231017_20k__1__binocular__lmdb.json'
        ]
    },
    'test_data': {
        'ella': [
            'data_hand/hand_keypoint/annotations/test_nreal_gesture_1111_1_1_twohand_gesture_lmdb.json'
        ],
        'flora_static_finegrain': [
            'data_hand/hand_keypoint/annotations/hand_test_flora_static_benchmark_230627_10k_lmdb.json',  # flora test
            'data_hand/hand_keypoint/annotations/hand_test_flora_static_benchmark_230703_10k_lmdb.json',  # flora test
            'data_hand/hand_keypoint/annotations/hand_test_flora_static_benchmark_230712_8k_lmdb.json'  # flora test
        ],
        'flora_dynamic': [
            'data_hand/hand_keypoint/annotations/hand_test_dynamic_keypoint_230907_20k__1__binocular__lmdb.json'
        ],
        'flora_black': [
            'data_hand/hand_keypoint/annotations/hand_test_chichi_keypoint_230912_5k__1__binocular__lmdb.json',
            'data_hand/hand_keypoint/annotations/hand_test_chichis_keypoint_230918_5k__1__binocular__lmdb.json'
        ],
        # 车内数据
        'flora_car': [
            'data_hand/hand_keypoint/annotations/hand_test_flora_keypoint_car_231208_2k__1__binocular__lmdb.json'
        ],
        # 手部佩戴装饰品数据
        'flora_decoration': [
            'data_hand/hand_keypoint/annotations/hand_test_flora_keypoint_decoration_1_231208_1k__1__binocular__lmdb.json',
            'data_hand/hand_keypoint/annotations/hand_test_flora_keypoint_decoration_2_231208_1k__1__binocular__lmdb.json',
            'data_hand/hand_keypoint/annotations/hand_test_flora_keypoint_decoration_3_231208_1k__1__binocular__lmdb.json',
        ]
    }
}
