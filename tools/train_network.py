import argparse
import datetime
import os

import numpy as np
import torch
import torchvision

from sudoku.network import (
    get_model, 
    init_weights, 
    Mean, 
    checkpoint_path,
    save_weights, 
    load_weights, 
    TimeEstimator
)

# ------------------------------------------------------------------------------

def transform(img) -> np.ndarray:

    '''
    Prepare raw images for training.
    
    Arguments:
        x (PIL.Image.Image): Image from EMNIST dataset.
        
    Returns: 
        (ndarray<float32>): 1x28x28 grayscale image in the range [0, 1].
    '''

    x = np.array(img, dtype=np.float32) / 255.    ## Normalize to range [0, 1]
    x = np.fliplr(x)            ## EMNIST dataset has incorrect orientation
    x = np.rot90(x)
    x = np.expand_dims(x, 0)
    return x

# ------------------------------------------------------------------------------

def _parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='./', help='directory to the EMNIST dataset. Will be downloaded if not present')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/', help='directory to save model checkpoints')
    parser.add_argument('--checkpoint-interval', type=int, default=1, help='save a checkpoint every X epochs')
    parser.add_argument('--use-cuda', type=bool, default=True, help='use CUDA, if available')
    parser.add_argument('--cudnn-benchmark', type=bool, default=True, help='use cudnn benchmark')
    parser.add_argument('--num-workers', type=int, default=4, help='number of cpu workers for DataLoader')
    
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--resume-epoch', type=int, default=0, help='if non-zero, resume training from X epoch')
    parser.add_argument('--batch-size', type=int, default=8, help='training batch size')
    parser.add_argument('--shuffle', type=bool, default=True, help='shuffle training data')
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9, help='Adam: beta 1')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam: beta 2')
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='Adam: weight decay')
    parser.add_argument('--lr-decay-step', type=int, default=5, help='decay learning rate every X epochs')
    parser.add_argument('--lr-decay-factor', type=float, default=0.5, help='factor to decay learning rate')
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = _parse_args()
    for k, v in vars(args).items():
        print(f'--{k}={v}')

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # check cuda
    use_cuda = torch.cuda.is_available() and args.use_cuda
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print(f'Training on device: {device}')

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    
    # load model
    model = get_model()
    
    # load weights
    if args.resume_epoch != 0:
        print(f'Resuming from epoch {args.resume_epoch}')
        path_to_weights = checkpoint_path(args.checkpoint_dir, args.resume_epoch)
        load_weights(model, path_to_weights)
    else:
        print('Initializing new weights')
        model.apply(init_weights)
    model = model.to(device)

    # load datasets
    train_dataset = torchvision.datasets.EMNIST(
        args.dataset_dir, 
        split='digits',
        train=True, 
        download=True,
        transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=args.shuffle, 
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last=True)
    val_dataset = torchvision.datasets.EMNIST(
        args.dataset_dir, 
        split='digits',
        train=False, 
        download=True,
        transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)

    print(f'Batch size: {args.batch_size}')
    print(f'Number of training batches: {len(train_loader)}')
    print(f'Number of validation batches: {len(val_loader)}')

    # load optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.b1, args.b2), 
        weight_decay=args.weight_decay)

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # --------------------------------------------------------------------------

    timeEstimator = TimeEstimator((args.epochs - args.resume_epoch) * len(train_loader))

    for epoch in range(args.resume_epoch, args.epochs):

        # adjust learning rate
        lr = args.lr * (args.lr_decay_factor ** (epoch // args.lr_decay_step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('INFO | Epoch {}/{} | lr {:.2e}'.format(epoch+1, args.epochs, lr))

        # training step
        timeEstimator.reset()
        model.train()
        for batch_i, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(image)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

            delta_t, remaining_t = timeEstimator.update()
            print('TRAIN | Epoch {}/{} | Batch {}/{} | Loss {:.4f} | {:.2f} sec | {} remaining'.format(
                epoch+1, args.epochs, batch_i+1, len(train_loader), loss.item(), 
                delta_t, datetime.timedelta(seconds=remaining_t)))

        # initialize metrics
        mean_loss = Mean()
        accuracy = Mean()
        
        # validation 
        model.eval()
        for batch_i, (image, label) in enumerate(val_loader):
            image = image.to(device)
            label = label.to(device)

            with torch.no_grad():
                out = model(image)
                loss = criterion(out, label)

            out = out.cpu().detach()
            loss = loss.cpu().detach()
            label = label.cpu().detach()

            print('VAL | Epoch {}/{} | Batch {}/{} | Loss {:.4f}'.format(
                epoch+1, args.epochs, batch_i+1, len(val_loader), loss.item()))

            # update metrics
            pred = out.argmax(1)
            mean_loss.accumulate(loss)
            accuracy.accumulate(pred == label)

        # print metrics
        print('METRIC | Epoch {}/{} | Mean Loss {:.4f} | Accuracy {:.4f}'.format(
            epoch+1, args.epochs, mean_loss.result().item(), accuracy.result().item()))
    
        # save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            path_to_weights = checkpoint_path(args.checkpoint_dir, epoch+1)
            save_weights(model, path_to_weights)

    total_t = timeEstimator.total()
    print(f'Total Elapsed Time: {datetime.timedelta(seconds=total_t)}')
