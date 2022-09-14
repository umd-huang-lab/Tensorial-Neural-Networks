"""

This is the mixed precision implementation of ResNet Code

Reference https://github.com/NVIDIA/apex


"""

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np

from networks import *
from tensorboardX import SummaryWriter

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


def main():

    global best_prec1, args

    args = parse()

    if(args.local_rank == 0):
        print("decomposition type: {}".format(args.decompose_type))
        print("batch size is {}".format(args.batch_size))
        print("compression rate is {}".format(args.compression_rate))


    cudnn.benchmark = True
    best_prec1 = 0

    if args.distributed:

        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    # model
    if(args.local_rank == 0):
        print("Model: ", args.model)
    model = ResNet_(args.model, args.decompose_type, args.compression_rate)

    model = model.cuda().to(memory_format=memory_format)

    # Scale learning rate based on global batch size
    batch_size  = args.batch_size

    args.learning_rate = args.learning_rate*float(args.batch_size*args.world_size)/256.

    # optimizer and corresponding scheduler
    if args.sgd:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
         optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)


    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
        resume()


     ## Paths (Dataset, Checkpoints, Statistics and TensorboardX)
    data_path = args.data_path

    # Data loading

    batch_size  = args.batch_size
    # number of image channels (1), and image height/width (2, 3)
    image_height   = args.image_height
    image_width    = args.image_width
    image_channels = args.image_channels

     # preprocessing/transformation of the input images
    image_padding = args.image_padding

    # number of worker for dataloaders
    num_workers = 4

    # data augmentation
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    if image_channels == 1:

        train_transform = transforms.Compose(
            [transforms.Resize((image_height, image_width)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomCrop((image_height, image_width), padding = image_padding),
             transforms.Grayscale(num_output_channels = image_channels),
             transforms.ToTensor(),
             normalize])

        valid_transform = transforms.Compose(
            [transforms.Resize((image_height, image_width)),
             transforms.Grayscale(num_output_channels = image_channels),
             transforms.ToTensor(),
             normalize])

    else:

        train_transform = transforms.Compose(
            [transforms.Resize((image_height, image_width)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomCrop((image_height, image_width), padding = image_padding),
             transforms.ToTensor(),
             normalize])

        valid_transform = transforms.Compose(
            [transforms.Resize((image_height, image_width)),
             transforms.ToTensor(),
             normalize])

    # training, validation and test sets
    Dataset = datasets.CIFAR10
    train_dataset = Dataset(root = data_path, train = True,
        download = False,transform=train_transform)
    valid_dataset = Dataset(root = data_path, train = False,
        download = False,transform=valid_transform)



    train_sampler = None
    valid_sampler = None

    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=valid_sampler)



    train_samples = len(train_loader) * batch_size
    valid_samples = len(valid_loader) * batch_size

    if(args.local_rank == 0):
        print("# of training samples: ", train_samples)
        print("# of validation samples: ", valid_samples)

    train_time = 0.0
    for epoch in range(args.start_epoch, args.epoch_num):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch

        #train(train_loader, model, optimizer, epoch, tensorboard_writer)
        train(train_loader, model, optimizer, epoch, 0)
        # evaluate on validation set
        #prec1 = validate(valid_loader, model,tensorboard_writer)
        prec1 = validate(valid_loader, model, 0)




        # remember best prec@1 and save checkpoint
        # if args.local_rank == 0 and epoch % 10 == 0:
        #     is_best = prec1 > best_prec1
        #     best_prec1 = max(prec1, best_prec1)
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'best_prec1': best_prec1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best, str(args.compression_rate) + "_" + str(epoch))




class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.4914 * 255, 0.4914 * 255, 0.4465 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.2023 * 255, 0.1994* 255, 0.2010 * 255]).cuda().view(1,3,1,1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target


def train(train_loader, model, optimizer, epoch, tensorboard_writer):
    """
        Model training

    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    # switch to train mode
    model.train()
    end = time.time()
    total_start = time.time()

    prefetcher = data_prefetcher(train_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1
        if args.prof >= 0 and i == args.prof:
            print("Profiling begun at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStart()

        if args.prof >= 0: torch.cuda.nvtx.range_push("Body of iteration {}".format(i))

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        # compute output
        if args.prof >= 0: torch.cuda.nvtx.range_push("forward")
        output = model(input)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()
        loss = F.nll_loss(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.prof >= 0: torch.cuda.nvtx.range_push("backward")
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # for param in model.parameters():
        #     print(param.data.double().sum().item(), param.grad.data.double().sum().item())

        if args.prof >= 0: torch.cuda.nvtx.range_push("optimizer.step()")
        optimizer.step()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()



        if i%args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Measure accuracy
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(loss.data)
                prec1 = reduce_tensor(prec1)
                prec5 = reduce_tensor(prec5)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end)/args.print_freq)
            end = time.time()


            if args.local_rank == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                # tensorboard_writer.add_scalar('lr', learning_rate, epoch)
                # tensorboard_writer.add_scalar('train_nll', losses.val, i*args.batch_size)
                # tensorboard_writer.add_scalar('train_top1', top1.val, i*args.batch_size)
                # tensorboard_writer.add_scalar('train_top5', top5.val, i*args.batch_size)
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Speed {3:.3f} ({4:.3f})\t'
                      'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader),
                       args.world_size*args.batch_size/batch_time.val,
                       args.world_size*args.batch_size/batch_time.avg,
                       batch_time=batch_time,
                       loss=losses,
                       top1=top1,
                       top5=top5))
        if args.prof >= 0: torch.cuda.nvtx.range_push("prefetcher.next()")
        input, target = prefetcher.next()
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        # Pop range "Body of iteration {}".format(i)
        if args.prof >= 0: torch.cuda.nvtx.range_pop()

        if args.prof >= 0 and i == args.prof + 10:
            print("Profiling ended at iteration {}".format(i))
            torch.cuda.cudart().cudaProfilerStop()
            quit()
    torch.cuda.synchronize()
    if args.local_rank == 0:
        print("[Training] Epoch:{0} total_time: {1:.3f}".format(
            epoch, time.time()-total_start))



def validate(val_loader, model, tensorboard_writer):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    total_start=time.time()
    end = time.time()

    prefetcher = data_prefetcher(val_loader)
    input, target = prefetcher.next()
    i = 0
    while input is not None:
        i += 1

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = F.nll_loss(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        losses.update(to_python_float(reduced_loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # TODO:  Change timings to mirror train().
        if args.local_rank == 0 and i % args.print_freq == 0:
            # tensorboard_writer.add_scalar('validate_nll', losses.val, i*args.batch_size)
            # tensorboard_writer.add_scalar('validate_top1', top1.val, i*args.batch_size)
            # tensorboard_writer.add_scalar('validate_top5', top5.val, i*args.batch_size)
            
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Speed {2:.3f} ({3:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader),
                   args.world_size * args.batch_size / batch_time.val,
                   args.world_size * args.batch_size / batch_time.avg,
                   batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

        input, target = prefetcher.next()

    torch.cuda.synchronize()
    if args.local_rank == 0:
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\t'
              '[Test] total_time: {total_time:.3f}'
              .format(top1=top1, top5=top5, total_time=time.time()-total_start))

    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.learning_rate*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt





def parse():
    parser = argparse.ArgumentParser(description =
        "CIFAR10 experiment on ResNet with tensor decomposition")

    ## Data format (Pytorch format)
    # batch size (0) x channels (1) x height (2) x width (3)
    parser.add_argument("--batch-size",  default = 128,  type = int,
        help = "The batch size for training.")





    # image format: channels (1), height (2), width (3)
    parser.add_argument("--image-height", default =  32, type = int,
        help = "The image height of each sample.")
    parser.add_argument("--image-width",  default =  32, type = int,
        help = "The image width  of each sample.")
    parser.add_argument("--image-channels", default = 3, type = int,
        help = "The number of channels in each sample.")


    ## Paths (Data, Checkpoints, Results and TensorboardX)

    # inputs:  data
    parser.add_argument("--dataset", default = "CIFAR10", type = str,
        help = "The dataset used for training.")
    parser.add_argument("--data-path", default = "./data/CIFAR10", type = str,
        help = "The path to the folder stroing the data.")


    # outputs: checkpoints, statistics and tensorboard
    parser.add_argument("--outputs-path", default = "/cmlscratch/xliu1231/experiment/CIFAR10/outputs", type = str,
        help = "The path to the folder storing outputs from training.")
    parser.add_argument("--model", default="ResNet34", type=str,
        help = "The name of the model.")
    parser.add_argument("--model-stamp", default = "default", type = str,
        help = "The time stamp of the model (as a suffix to the its name).")
    parser.add_argument("--model-path", default = "models", type = str,
        help = "The folder for all checkpoints in training.")
    parser.add_argument("--stats-path", default = "stats",  type = str,
        help = "The folder for the evaluation statistics.")
    parser.add_argument("--stats-file", default = "curve",  type = str,
        help = "The file name for the learning curve.")
    parser.add_argument('--tensorboard-path', default = 'tensorboard', type = str,
        help = 'The folder for the tensorboardX files.')

     # parameters for DDP
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--prof', default=-1, type=int,
                        help='Only run 10 iterations for profiling.')

    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--distributed', type=bool, default=True)
    parser.add_argument('--evaluate', type=bool, default=True)


    parser.add_argument('--compression-rate', type=float, default=1)
    parser.add_argument('--decompose-type', type=str, default=None)

    #resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    #
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--epoch-num', default=100, type=int)
    parser.add_argument("--train-ratio", default = 0.9, type=float,
        help="The ratio of training samples in the .")
    parser.add_argument("--print_freq", default = 10, type = int,
        help = "Log the learning curve every print_freq iterations.")


    # learning rate scheduling
    parser.add_argument("--learning-rate", default = 0.3, type = float,
        help = "Initial learning rate of the optimizer.")

    parser.add_argument("--learning-rate-decay", dest = "rate_decay", action = 'store_true',
        help = "Learning rate is decayed during training.")
    parser.add_argument("--learning-rate-fixed", dest = "rate_decay", action = 'store_false',
        help = "Learning rate is fixed during training.")
    parser.set_defaults(rate_decay = True)

    # if rate_decay is True (--learning-rate-decay)
    parser.add_argument("--decay-epoch", default = "30", type = str,
        help = "The learning rate is decayed by decay_rate every decay_epoch.")
    parser.add_argument("--decay-rate", default = 0.5, type = float,
        help = "The learning rate is decayed by decay_rate every decay_epoch.")

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.set_defaults(sgd=True)
    # data augmentation (in learning phase)
    parser.add_argument("--image-padding",  default = 4, type = int,
        help = "The number of padded pixels along height/width.")






    return parser.parse_args()



if __name__ == "__main__":
    main()
