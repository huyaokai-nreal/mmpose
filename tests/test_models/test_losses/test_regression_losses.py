# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmpose.models.losses.regression_loss import (PinchLoss,
                                                  SoftWeightSmoothL1Loss)


class TestSoftWeightSmoothL1Loss(TestCase):

    def test_loss(self):

        # test loss w/o target_weight
        loss = SoftWeightSmoothL1Loss(use_target_weight=False, beta=0.5)

        fake_pred = torch.zeros((1, 3, 2))
        fake_label = torch.zeros((1, 3, 2))
        self.assertTrue(
            torch.allclose(loss(fake_pred, fake_label), torch.tensor(0.)))

        fake_pred = torch.ones((1, 3, 2))
        fake_label = torch.zeros((1, 3, 2))
        self.assertTrue(
            torch.allclose(loss(fake_pred, fake_label), torch.tensor(.75)))

        # test loss w/ target_weight
        loss = SoftWeightSmoothL1Loss(
            use_target_weight=True, supervise_empty=True)

        fake_pred = torch.ones((1, 3, 2))
        fake_label = torch.zeros((1, 3, 2))
        fake_weight = torch.arange(6).reshape(1, 3, 2).float()
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label, fake_weight), torch.tensor(1.25)))

        # test loss that does not take empty channels into account
        loss = SoftWeightSmoothL1Loss(
            use_target_weight=True, supervise_empty=False)
        self.assertTrue(
            torch.allclose(
                loss(fake_pred, fake_label, fake_weight), torch.tensor(1.5)))

        with self.assertRaises(ValueError):
            _ = loss.smooth_l1_loss(fake_pred, fake_label, reduction='fake')

        output = loss.smooth_l1_loss(fake_pred, fake_label, reduction='sum')
        self.assertTrue(torch.allclose(output, torch.tensor(3.0)))


class TestPinchLoss(TestCase):

    def test_loss(self):

        test_num = 10000

        enter_thre, exit_thre, loss_weight = 0.02, 0.04, 1
        loss = PinchLoss(
            enter_thre=enter_thre,
            exit_thre=exit_thre,
            loss_weight=loss_weight)

        # gt. loss = 0
        gt_dist = torch.rand(test_num).unsqueeze(1)
        loss_list = [loss(gt_dist[i], gt_dist[i]) for i in range(test_num)]
        self.assertTrue(all(item == 0 for item in loss_list))

        # gt is pinch and pred is pinch, or gt isn't pinch and pred isn't pinch. loss = 0
        gt_dist = torch.cat(
            ((torch.rand(test_num // 2) * enter_thre).unsqueeze(1),
             (torch.rand(test_num // 2) + (exit_thre + 0.0001)).unsqueeze(1)),
            dim=0)
        pred_dist = torch.cat(
            ((torch.rand(test_num // 2) * enter_thre).unsqueeze(1),
             (torch.rand(test_num // 2) + (exit_thre + 0.0001)).unsqueeze(1)),
            dim=0)
        loss_list = [loss(pred_dist[i], gt_dist[i]) for i in range(test_num)]
        self.assertTrue(all(item == 0 for item in loss_list))

        # gt is pinch, but pred isn't pinch. loss > 0
        gt_dist = torch.rand(test_num).unsqueeze(1) * enter_thre
        pred_dist = torch.rand(test_num).unsqueeze(1) + enter_thre
        loss_list = [loss(pred_dist[i], gt_dist[i]) for i in range(test_num)]
        self.assertTrue(all(item > 0 for item in loss_list))

        # gt isn't pinch, but pred is pinch. loss > 0
        gt_dist = torch.rand(test_num).unsqueeze(1) + exit_thre
        pred_dist = torch.rand(test_num).unsqueeze(1) * exit_thre
        loss_list = [loss(pred_dist[i], gt_dist[i]) for i in range(test_num)]
        self.assertTrue(all(item > 0 for item in loss_list))
