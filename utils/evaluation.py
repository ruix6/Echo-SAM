from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
import torch.nn.functional as F
import utils.metrics as metrics
from hausdorff import hausdorff_distance
import time
import pandas as pd

def eval_mask_slice(val_loader, model, criterion, mode, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = []
    hds = []
    ious, accs, ses, sps = [], [], [], []
    eval_number = 0
    sum_time = 0
    pred_all = []
    for batch_idx, (datapack) in enumerate(val_loader):
        imgs = datapack['image'].to(dtype = torch.float32, device=args.device)
        masks = datapack['mask'].to(dtype = torch.float32, device=args.device)
        #bbox = datapack['bbox'].to(dtype=torch.float32, device=args.device)
        points_prompts = datapack['points_prompts'].to(dtype=torch.float32, device=args.device)
        points_labels = datapack['points_labels'].to(dtype=torch.float32, device=args.device)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, None, None, points_prompts, points_labels)#, None, None)
            sum_time =  sum_time + (time.time()-start_time)
            pred_all.append(pred)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        gt = masks.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred)
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices.append(dice_i)
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious.append(iou)
            accs.append(acc)
            ses.append(se)
            sps.append(sp)
            hds.append(hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan"))
            del pred_i, gt_i
        eval_number = eval_number + b

    mean_dice = np.mean(dices)
    std_dice = np.std(dices)
    mean_hds = np.mean(hds)
    std_hds = np.std(hds)
    mean_iou, mean_acc, mean_se, mean_sp = np.mean(ious), np.mean(accs), np.mean(ses), np.mean(sps)
    std_iou, std_acc, std_se, std_sp = np.std(ious), np.std(accs), np.std(ses), np.std(sps)

    val_losses = val_losses / (batch_idx + 1)

    print("test speed", eval_number/sum_time)
    if mode == "train":
        del pred_all
        return dices, mean_dice, mean_hds, val_losses
    elif mode == "test":
        del pred_all
        return mean_dice, mean_hds, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hds, std_iou, std_acc, std_se, std_sp

def eval_mask_slice_compare(val_loader, model, criterion, mode, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = []
    hds = []
    ious, accs, ses, sps = [], [], [], []
    eval_number = 0
    sum_time = 0
    pred_all = []
    for batch_idx, (datapack) in enumerate(val_loader):
        imgs = datapack['image'].to(dtype = torch.float32, device=args.device)
        masks = datapack['mask'].to(dtype = torch.float32, device=args.device)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs)
            sum_time =  sum_time + (time.time()-start_time)
            pred_all.append(pred)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        gt = masks.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred)
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices.append(dice_i)
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious.append(iou)
            accs.append(acc)
            ses.append(se)
            sps.append(sp)
            hds.append(hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan"))
            del pred_i, gt_i
        eval_number = eval_number + b

    mean_dice = np.mean(dices)
    std_dice = np.std(dices)
    mean_hds = np.mean(hds)
    std_hds = np.std(hds)
    mean_iou, mean_acc, mean_se, mean_sp = np.mean(ious), np.mean(accs), np.mean(ses), np.mean(sps)
    std_iou, std_acc, std_se, std_sp = np.std(ious), np.std(accs), np.std(ses), np.std(sps)

    val_losses = val_losses / (batch_idx + 1)

    print("test speed", eval_number/sum_time)
    if mode == "train":
        del pred_all
        return dices, mean_dice, mean_hds, val_losses
    elif mode == "test":
        del pred_all
        return mean_dice, mean_hds, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hds, std_iou, std_acc, std_se, std_sp
    

def eval_mask_sam(val_loader, model, criterion, mode, args):
    model.eval()
    val_losses, mean_dice = 0, 0
    dices = []
    hds = []
    ious, accs, ses, sps = [], [], [], []
    eval_number = 0
    sum_time = 0
    pred_all = []
    for batch_idx, (datapack) in enumerate(val_loader):
        imgs = datapack['image'].to(dtype = torch.float32, device=args.device)
        masks = datapack['mask'].to(dtype = torch.float32, device=args.device)
        bbox = datapack['bbox'].to(dtype=torch.float32, device=args.device)
        points_prompts = datapack['points_prompts'].to(dtype=torch.float32, device=args.device)
        points_labels = datapack['points_labels'].to(dtype=torch.float32, device=args.device)

        with torch.no_grad():
            start_time = time.time()
            pred = model(imgs, None, bbox, points_prompts, points_labels)
            sum_time =  sum_time + (time.time()-start_time)
            pred_all.append(pred)

        val_loss = criterion(pred, masks)
        val_losses += val_loss.item()

        gt = masks.detach().cpu().numpy()
        gt = gt[:, 0, :, :]
        predict = torch.sigmoid(pred)
        predict = predict.detach().cpu().numpy()  # (b, c, h, w)
        seg = predict[:, 0, :, :] > 0.5  # (b, h, w)
        b, h, w = seg.shape
        for j in range(0, b):
            pred_i = np.zeros((1, h, w))
            pred_i[seg[j:j+1, :, :] == 1] = 255
            gt_i = np.zeros((1, h, w))
            gt_i[gt[j:j+1, :, :] == 1] = 255
            dice_i = metrics.dice_coefficient(pred_i, gt_i)
            #print("name:", name[j], "coord:", coords_torch[j], "dice:", dice_i)
            dices.append(dice_i)
            iou, acc, se, sp = metrics.sespiou_coefficient2(pred_i, gt_i, all=False)
            ious.append(iou)
            accs.append(acc)
            ses.append(se)
            sps.append(sp)
            hds.append(hausdorff_distance(pred_i[0, :, :], gt_i[0, :, :], distance="manhattan"))
            del pred_i, gt_i
        eval_number = eval_number + b

    mean_dice = np.mean(dices)
    std_dice = np.std(dices)
    mean_hds = np.mean(hds)
    std_hds = np.std(hds)
    mean_iou, mean_acc, mean_se, mean_sp = np.mean(ious), np.mean(accs), np.mean(ses), np.mean(sps)
    std_iou, std_acc, std_se, std_sp = np.std(ious), np.std(accs), np.std(ses), np.std(sps)

    val_losses = val_losses / (batch_idx + 1)

    print("test speed", eval_number/sum_time)
    if mode == "train":
        del pred_all
        return dices, mean_dice, mean_hds, val_losses
    elif mode == "test":
        del pred_all
        return mean_dice, mean_hds, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hds, std_iou, std_acc, std_se, std_sp