import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch
import random
from utils.evaluation import eval_mask_slice
from utils.loss_functions.sam_loss import get_criterion
from echo_sam import echo_sam_model_registry
ttt = 'Echo-SAM'
from utils.dataset import SegmentationDataset, SegmentationDatasetCAMUS
from torch.utils.data import DataLoader
import csv


def main():

    #  =========================================== parameters setting ==================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='Echo-SAM', type=str, help='type of model')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size per gpu') 
    parser.add_argument('--device', type=str, default='cuda', help='device for training')

    checkpoint_path = 'work_dir/Echo_SAM/Echo_SAM_02100126.pth'

    args = parser.parse_args()
    print("task", args.task, "checkpoints:", checkpoint_path)
    mode = "test"
    device = torch.device(args.device)

    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 300 # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model

    echo_sam_model = echo_sam_model_registry[args.vit_name](checkpoint=checkpoint_path)
    echo_sam_model.to(device)
    echo_sam_model.eval()
    print(echo_sam_model.image_encoder.alpha)

    with open('result-Echo-SAM.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['mode', 'task', 'mean-Dice', 'std-Dice', 'mean-HDS', 'std-HDS', 'mean-IoU', 
                         'std-IoU', 'mean-Acc', 'std-Acc', 'mean-SE', 'std-SE', 'mean-SP', 'std-SP'])

        for task in ['HMC-QU', 'EchoNet']:


            dataset_test = SegmentationDataset(task, mode, num_points=1, aug=False, size=256)
            test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

            criterion = get_criterion(modelname=args.modelname)
            mean_dice, mean_hds, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hds, std_iou, std_acc, std_se, std_sp = eval_mask_slice(test_loader, echo_sam_model, criterion=criterion, mode='test', args=args)
            print("dataset:" + task + " -----------model name: "+ args.modelname)
            print(f'Dice: {mean_dice:.4f} +- {std_dice:.4f}')
            print(f'HDS: {mean_hds:.4f} +- {std_hds:.4f}')
            print(f'IoU: {mean_iou:.4f} +- {std_iou:.4f}')
            print(f'Acc: {mean_acc:.4f} +- {std_acc:.4f}')
            print(f'SE: {mean_se:.4f} +- {std_se:.4f}')
            print(f'SP: {mean_sp:.4f} +- {std_sp:.4f}')

            writer.writerow([ttt, task, mean_dice, std_dice, mean_hds, std_hds, mean_iou, std_iou, mean_acc, std_acc, mean_se, std_se, mean_sp, std_sp])

        for pos in ['LA', 'LV', 'MYO']:


            dataset_test = SegmentationDatasetCAMUS(data_dir='CAMUS', pos=pos, mode='test', num_points=1, size=256)
            test_loader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

            criterion = get_criterion(modelname=args.modelname)
            mean_dice, mean_hds, mean_iou, mean_acc, mean_se, mean_sp, std_dice, std_hds, std_iou, std_acc, std_se, std_sp = eval_mask_slice(test_loader, echo_sam_model, criterion=criterion, mode='test', args=args)
            print("dataset:" + task + " -----------model name: "+ args.modelname)
            print(f'Dice: {mean_dice:.4f} +- {std_dice:.4f}')
            print(f'HDS: {mean_hds:.4f} +- {std_hds:.4f}')
            print(f'IoU: {mean_iou:.4f} +- {std_iou:.4f}')
            print(f'Acc: {mean_acc:.4f} +- {std_acc:.4f}')
            print(f'SE: {mean_se:.4f} +- {std_se:.4f}')
            print(f'SP: {mean_sp:.4f} +- {std_sp:.4f}')

            writer.writerow([ttt, pos, mean_dice, std_dice, mean_hds, std_hds, mean_iou, std_iou, mean_acc, std_acc, mean_se, std_se, mean_sp, std_sp])


if __name__ == '__main__':
    main()