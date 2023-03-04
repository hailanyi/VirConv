import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time

from dataloaders.kitti_loader import load_calib, input_options, KittiDepth
from metrics import AverageMeter, Result
import criteria
import helper
import vis_utils

from model import ENet
from model import PENet_C1_train
from model import PENet_C2_train
#from model import PENet_C4_train (Not Implemented)
from model import PENet_C1
from model import PENet_C2
from model import PENet_C4
import time

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-n',
                    '--network-model',
                    type=str,
                    default="pe",
                    choices=["e", "pe"],
                    help='choose a model: enet or penet'
                    )
parser.add_argument('--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run (default: 100)')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start-epoch-bias',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number bias(useful on restarts)')
parser.add_argument('-c',
                    '--criterion',
                    metavar='LOSS',
                    default='l2',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) +
                    ' (default: l2)')
parser.add_argument('-b',
                    '--batch-size',
                    default=1,
                    type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=1e-3,
                    type=float,
                    metavar='LR',
                    help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-6,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 0)')
parser.add_argument('--print-freq',
                    '-p',
                    default=10,
                    type=int,
                    metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--data-folder',
                    default='/data/dataset/kitti_depth/depth',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
parser.add_argument('--data-folder-rgb',
                    default='/data/dataset/kitti_raw',
                    type=str,
                    metavar='PATH',
                    help='data folder rgb (default: none)')
parser.add_argument('--data-folder-save',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='data folder test results(default: none)')
parser.add_argument('--detpath',
                    default='../../data/kitti/training',
                    type=str,
                    metavar='PATH',
                    help='data folder of 3D object detection')
parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='rgbd',
                    choices=input_options,
                    help='input: | '.join(input_options))
parser.add_argument('--val',
                    type=str,
                    default="select",
                    choices=["select", "full"],
                    help='full or select validation set')
parser.add_argument('--jitter',
                    type=float,
                    default=0.1,
                    help='color jitter for images')
parser.add_argument('--rank-metric',
                    type=str,
                    default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help='metrics for which best result is saved')

parser.add_argument('-e', '--evaluate', default='pe.pth.tar', type=str, metavar='PATH')
parser.add_argument('-f', '--freeze-backbone', action="store_true", default=False,
                    help='freeze parameters in backbone')
parser.add_argument('--test', action="store_true", default=True,
                    help='save result kitti test dataset for submission')
parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

#random cropping
parser.add_argument('--not-random-crop', action="store_true", default=False,
                    help='prohibit random cropping')
parser.add_argument('-he', '--random-crop-height', default=320, type=int, metavar='N',
                    help='random crop height')
parser.add_argument('-w', '--random-crop-width', default=1216, type=int, metavar='N',
                    help='random crop height')

#geometric encoding
parser.add_argument('-co', '--convolutional-layer-encoding', default="xyz", type=str,
                    choices=["std", "z", "uv", "xyz"],
                    help='information concatenated in encoder convolutional layers')

#dilated rate of DA-CSPN++
parser.add_argument('-d', '--dilation-rate', default="2", type=int,
                    choices=[1, 2, 4],
                    help='CSPN++ dilation rate')

args = parser.parse_args()
args.result = os.path.join('..', 'results')
args.use_rgb = ('rgb' in args.input)
args.use_d = 'd' in args.input
args.use_g = 'g' in args.input
args.val_h = 352
args.val_w = 1216
print(args)

cuda = torch.cuda.is_available() and not args.cpu
if cuda:
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("=> using '{}' for computation.".format(device))

# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (
    args.criterion == 'l2') else criteria.MaskedL1Loss()

#multi batch
multi_batch_size = 1
def iterate(mode, args, loader, model, optimizer, logger, epoch):
    actual_epoch = epoch - args.start_epoch + args.start_epoch_bias

    block_average_meter = AverageMeter()
    block_average_meter.reset(False)
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)
    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, actual_epoch, args)
    else:
        model.eval()
        lr = 0

    torch.cuda.empty_cache()

    for i, batch_data in enumerate(loader):

        dstart = time.time()
        batch_data = {
            key: val.to(device)
            for key, val in batch_data.items() if val is not None
        }

        gt = batch_data[
            'gt'] if mode != 'test_prediction' and mode != 'test_completion' else None
        data_time = time.time() - dstart

        pred = None
        start = None
        gpu_time = 0

        #start = time.time()
        #pred = model(batch_data)
        #gpu_time = time.time() - start

        #'''
        if(args.network_model == 'e'):
            start = time.time()
            st1_pred, st2_pred, pred = model(batch_data)
        else:
            start = time.time()
            pred = model(batch_data)

        if(args.evaluate):
            gpu_time = time.time() - start
        #'''

        depth_loss, photometric_loss, smooth_loss, mask = 0, 0, 0, None

        # inter loss_param
        st1_loss, st2_loss, loss = 0, 0, 0
        w_st1, w_st2 = 0, 0
        round1, round2, round3 = 1, 3, None
        if(actual_epoch <= round1):
            w_st1, w_st2 = 0.2, 0.2
        elif(actual_epoch <= round2):
            w_st1, w_st2 = 0.05, 0.05
        else:
            w_st1, w_st2 = 0, 0

        if mode == 'train':
            # Loss 1: the direct depth supervision from ground truth label
            # mask=1 indicates that a pixel does not ground truth labels
            depth_loss = depth_criterion(pred, gt)

            if args.network_model == 'e':
                st1_loss = depth_criterion(st1_pred, gt)
                st2_loss = depth_criterion(st2_pred, gt)
                loss = (1 - w_st1 - w_st2) * depth_loss + w_st1 * st1_loss + w_st2 * st2_loss
            else:
                loss = depth_loss

            if i % multi_batch_size == 0:
                optimizer.zero_grad()
            loss.backward()

            if i % multi_batch_size == (multi_batch_size-1) or i==(len(loader)-1):
                optimizer.step()
            print("loss:", loss, " epoch:", epoch, " ", i, "/", len(loader))

        if mode == "test_completion":

            vis_utils.save_depth_as_points(pred, i, args.detpath)

        if(not args.evaluate):
            gpu_time = time.time() - start
        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()
            if mode != 'test_prediction' and mode != 'test_completion':
                result.evaluate(pred.data, gt.data, photometric_loss)
                [
                    m.update(result, gpu_time, data_time, mini_batch_size)
                    for m in meters
                ]

                if mode != 'train':
                    logger.conditional_print(mode, i, epoch, lr, len(loader),
                                     block_average_meter, average_meter)
                logger.conditional_save_img_comparison(mode, i, batch_data, pred,
                                                   epoch)
                logger.conditional_save_pred(mode, i, pred, epoch)
        end_time = time.time()-dstart
        print('iter: ', i,'  ',  'remain time:', (len(loader)-i)*end_time//60, 'min')
    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best

def main():
    global args
    checkpoint = None
    is_eval = False
    if args.evaluate:
        args_new = args
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}' ... ".format(args.evaluate),
                  end='')
            checkpoint = torch.load(args.evaluate, map_location=device)
            #args = checkpoint['args']
            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            is_eval = True

            print("Completed.")
        else:
            is_eval = True
            print("No model found at '{}'".format(args.evaluate))
            #return

    elif args.resume:  # optionally resume from a checkpoint
        args_new = args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}' ... ".format(args.resume),
                  end='')
            checkpoint = torch.load(args.resume, map_location=device)

            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            print("Completed. Resuming from epoch {}.".format(
                checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer ... ", end='')
    model = None
    penet_accelerated = False
    if (args.network_model == 'e'):
        model = ENet(args).to(device)
    elif (is_eval == False):
        if (args.dilation_rate == 1):
            model = PENet_C1_train(args).to(device)
        elif (args.dilation_rate == 2):
            model = PENet_C2_train(args).to(device)
        elif (args.dilation_rate == 4):
            model = PENet_C4(args).to(device)
            penet_accelerated = True
    else:
        if (args.dilation_rate == 1):
            model = PENet_C1(args).to(device)
            penet_accelerated = True
        elif (args.dilation_rate == 2):
            model = PENet_C2(args).to(device)
            penet_accelerated = True
        elif (args.dilation_rate == 4):
            model = PENet_C4(args).to(device)
            penet_accelerated = True

    if (penet_accelerated == True):
        model.encoder3.requires_grad = False
        model.encoder5.requires_grad = False
        model.encoder7.requires_grad = False

    model_named_params = None
    model_bone_params = None
    model_new_params = None
    optimizer = None

    if checkpoint is not None:
        #print(checkpoint.keys())
        if (args.freeze_backbone == True):
            model.backbone.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint['model'], strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
        del checkpoint
    print("=> logger created.")

    test_dataset = None
    test_loader = None
    if (args.test):
        test_dataset = KittiDepth('test_completion', args)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True)
        iterate("test_completion", args, test_loader, model, None, logger, 0)
        return

    val_dataset = KittiDepth('val', args)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True)  # set batch size to be 1 for validation
    print("\t==> val_loader size:{}".format(len(val_loader)))

    if is_eval == True:
        for p in model.parameters():
            p.requires_grad = False

        result, is_best = iterate("val", args, val_loader, model, None, logger,
                              args.start_epoch - 1)
        return

    if (args.freeze_backbone == True):
        for p in model.backbone.parameters():
            p.requires_grad = False
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    elif (args.network_model == 'pe'):
        model_bone_params = [
            p for _, p in model.backbone.named_parameters() if p.requires_grad
        ]
        model_new_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        model_new_params = list(set(model_new_params) - set(model_bone_params))
        optimizer = torch.optim.Adam([{'params': model_bone_params, 'lr': args.lr / 10}, {'params': model_new_params}],
                                     lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    else:
        model_named_params = [
            p for _, p in model.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    print("completed.")

    model = torch.nn.DataParallel(model)

    # Data loading code
    print("=> creating data loaders ... ")
    if not is_eval:
        train_dataset = KittiDepth('train', args)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   sampler=None)
        print("\t==> train_loader size:{}".format(len(train_loader)))

    print("=> starting main loop ...")
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger, epoch)  # train for one epoch

        # validation memory reset
        for p in model.parameters():
            p.requires_grad = False
        result, is_best = iterate("val", args, val_loader, model, None, logger, epoch)  # evaluate on validation set

        for p in model.parameters():
            p.requires_grad = True
        if (args.freeze_backbone == True):
            for p in model.module.backbone.parameters():
                p.requires_grad = False
        if (penet_accelerated == True):
            model.module.encoder3.requires_grad = False
            model.module.encoder5.requires_grad = False
            model.module.encoder7.requires_grad = False

        helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_result': logger.best_result,
            'optimizer' : optimizer.state_dict(),
            'args' : args,
        }, is_best, epoch, logger.output_directory)


if __name__ == '__main__':
    main()