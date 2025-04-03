import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.evaluation import eval_mask_slice
from utils.loss_functions.sam_loss import get_criterion

from echo_sam import echo_sam_model_registry
from utils.common import print_trainable_parameters
from utils.dataset import SegmentationDatasetUS, SegmentationDatasetCAMUS
from torch.utils.data import DataLoader

def main():

    #  ============================================================================= parameters setting ====================================================================================

    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='Echo_SAM', type=str, help='type of model')
    parser.add_argument('-epochs', type=int, default=200, help='epochs')
    parser.add_argument('--task', default='CAMUS', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='./work_dir/MedSAM/medsam_vit_b.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size per gpu') 
    parser.add_argument('--n_gpu', type=int, default=4, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005, help='segmentation network learning rate')
    parser.add_argument('--warmup', type=bool, default=True, help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=25, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=True, help='keep the loss&lr&dice during training or not')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--save_path', type=str, default='./work_dir/Echo_SAM/', help='save path')
    parser.add_argument('--result_path', type=str, default='./result/Echo_SAM/', help='result path')
    parser.add_argument('--tensorboard_path', type=str, default='./tensorboard/Echo_SAM/', help='tensorboard path')
    parser.add_argument('--save_path_code', type=str, default='_', help='device for training')

    args = parser.parse_args()

    device = torch.device(args.device)
    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')  # initialize the tensorboard for record the training process
        boardpath = args.tensorboard_path + args.modelname + args.save_path_code + logtimestr
        if not os.path.isdir(boardpath):
            os.makedirs(boardpath)
        TensorWriter = SummaryWriter(boardpath)


    #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 1234  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    # Load the weights from MedSAM.
    medsam_state_dict = torch.load(args.sam_ckpt)

    # Init Echo-SAM.
    echo_sam_model = echo_sam_model_registry[args.vit_name](checkpoint=None)

    # Load Pre-train weights.
    for name, param in echo_sam_model.named_parameters():
        if name in medsam_state_dict and name !='image_encoder.pos_embed':
            param.data = medsam_state_dict[name].data
            param.requires_grad = False
        if 'mask_decoder' in name:
            param.requires_grad = True

    for name, param in echo_sam_model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    print_trainable_parameters(echo_sam_model)

    if args.n_gpu > 1:
        echo_sam_model = torch.nn.DataParallel(echo_sam_model)
        args.batch_size = args.batch_size * args.n_gpu

    dataset_train = SegmentationDatasetUS(mode='json/CUS-SAM/train-camus', num_points=1, chans=3, size=256)
    dataset_val = SegmentationDatasetUS(mode='json/CUS-SAM/val-camus', num_points=1, chans=3, size=256)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    echo_sam_model.to(device)

    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, echo_sam_model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(echo_sam_model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    criterion = get_criterion(modelname=args.modelname)

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    max_iterations = args.epochs * len(train_loader)
    best_dice, loss_log, dice_log = 0.0, np.zeros(args.epochs+1), np.zeros(args.epochs+1)
    for epoch in range(args.epochs):
        #  --------------------------------------------------------- training ---------------------------------------------------------
        echo_sam_model.train()
        train_losses = 0
        for batch_idx, (datapack) in enumerate(train_loader):
            imgs = datapack['image'].to(dtype = torch.float32, device=args.device)
            masks = datapack['mask'].to(dtype = torch.float32, device=args.device)
            points_prompts = datapack['points_prompts'].to(dtype=torch.float32, device=args.device)
            points_labels = datapack['points_labels'].to(dtype=torch.float32, device=args.device)
            # -------------------------------------------------------- forward --------------------------------------------------------
            pred = echo_sam_model(imgs, None, None, points_prompts, points_labels)
            train_loss = criterion(pred, masks) 
            # -------------------------------------------------------- backward -------------------------------------------------------
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
            print(train_loss)
            # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                    lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            iter_num = iter_num + 1

        #  -------------------------------------------------- log the train progress --------------------------------------------------
        print('epoch [{}/{}], train loss:{:.4f}'.format(epoch, args.epochs, train_losses / (batch_idx + 1)))
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', train_losses / (batch_idx + 1), epoch)
            TensorWriter.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = train_losses / (batch_idx + 1)

        #  --------------------------------------------------------- evaluation ----------------------------------------------------------
        if epoch % 5 == 0:
            echo_sam_model.eval()
            _, mean_dice, _, val_losses = eval_mask_slice(val_loader, echo_sam_model, criterion=criterion, mode='train', args=args)
            print('epoch [{}/{}], val loss:{:.4f}'.format(epoch, args.epochs, val_losses))
            print('epoch [{}/{}], val dice:{:.4f}'.format(epoch, args.epochs, mean_dice))
            if args.keep_log:
                TensorWriter.add_scalar('val_loss', val_losses, epoch)
                TensorWriter.add_scalar('dices', mean_dice, epoch)
                dice_log[epoch] = mean_dice
            if mean_dice > best_dice:
                best_dice = mean_dice
                timestr = time.strftime('%m%d%H%M')
                if not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                save_path = args.save_path + args.modelname + args.save_path_code + '%s' % timestr + '_' + str(epoch) + '_' + str(best_dice)
                torch.save(echo_sam_model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
        if epoch % 100 == 0 or epoch == (args.epochs-1):
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            save_path = args.save_path + args.modelname + args.save_path_code + '_' + str(epoch)
            torch.save(echo_sam_model.state_dict(), save_path + ".pth", _use_new_zipfile_serialization=False)
            if args.keep_log:
                with open(args.tensorboard_path + args.modelname + args.save_path_code + logtimestr + '/trainloss.txt', 'w') as f:
                    for i in range(len(loss_log)):
                        f.write(str(loss_log[i])+'\n')
                with open(args.tensorboard_path + args.modelname + args.save_path_code + logtimestr + '/dice.txt', 'w') as f:
                    for i in range(len(dice_log)):
                        f.write(str(dice_log[i])+'\n')

if __name__ == '__main__':
    main()