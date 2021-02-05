import argparse
import torch
import torch.nn as nn
import os
import warnings  

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.adni_classification.adniDataset import ADNIDataset
from src.adni_classification.model import ResNet, MLP
from src.adni_classification.utils import ConfusionMatrix


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np

device = torch.device('cuda')


def get_stat(prediction, gt):
    return {
        'TP': (prediction.bool() & gt.bool()).sum().item(),
        'TN': ((~prediction.bool()) & (~gt.bool())).sum().item(),
        'FP': (prediction.bool() & (~gt.bool())).sum().item(),
        'FN': ((~prediction.bool()) & gt.bool()).sum().item(),
    }


def get_stat_2(prediction, gt):
    return {
        'TP': (prediction == gt).sum().item(),
        'TN': 0,
        'FP': (prediction != gt).sum().item(),
        'FN': 0,
    }


def icl(args, mu, labels):
    diff = mu.unsqueeze(0) - mu.unsqueeze(1)
    diff = diff.norm(dim = -1)
    sq_loss = diff.pow(2)
    lap_kernel = torch.exp(args.icl_a - args.icl_b * diff)
    
    s = torch.zeros_like(labels).float()
    c = 3
    for i in range(c):
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            continue
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)
    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    loss = (sq_loss * s_prod)[c_diff == True].sum()
    loss += (c-1) * (lap_kernel * s_prod)[c_diff == False].sum()
    
    return loss


def mmd_loss(args, mu, labels):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.pow(2).sum(-1)

    s = torch.zeros_like(labels).float()
    c = 3
    for i in range(c):
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            continue
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)

    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    diff = diff * s_prod
    loss = diff[c_diff == True].sum()
    loss += -(c-1)*diff[c_diff == False].sum()
        
    return loss


def mmd_lap_loss(args, mu, labels):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.norm(dim = -1)
    lap_kernel = torch.exp(-diff * args.icl_b)

    s = torch.zeros_like(labels).float()
    c = 3
    for i in range(c):
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            continue
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)

    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    lap_kernel = lap_kernel * s_prod
    loss = -lap_kernel[c_diff == True].sum()
    loss += (c-1)*lap_kernel[c_diff == False].sum()
    
    return loss


def to_device(list_of_tensors, device):
    return [t.to(device) for t in list_of_tensors]


def main(args):
    train_dataset = ADNIDataset(ad_cn=True, split_filename='./data/splits/train_{:d}.data'.format(args.fold))
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True,
                                  num_workers=0, drop_last=False)

    valid_dataset = ADNIDataset(ad_cn=True, split_filename='./data/splits/val_{:d}.data'.format(args.fold))
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False, pin_memory=True,
                                  num_workers=0, drop_last=False)

    tensorboard_logdir = os.path.join(args.fold_dir, "tensorboard")
    if not os.path.exists(tensorboard_logdir):
        os.mkdir(tensorboard_logdir)
    writer = SummaryWriter(tensorboard_logdir)

    model = ResNet(1, args.blocks, args.channels, bottleneck=args.use_bottleneck_layers,
                   n_out_linear=1, dropout=0.5)
    print('Model used :', model)
    model = nn.DataParallel(model).to(device)

    disc_model = nn.DataParallel(MLP(args.channels[-1] * 2, [64] * 3, 3)).to(device)
    disc_optimizer = torch.optim.Adam(disc_model.parameters(), lr=args.lr)
    disc_loss_obj = nn.CrossEntropyLoss()

    loss_obj = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_cm = ConfusionMatrix()
    valid_cm = ConfusionMatrix()
    best_valid_loss = np.inf
    best_valid_acc = 0.

    for epoch in range(args.max_epochs):
        # Train loop
        if args.alpha < args.alpha_max:
            args.alpha *= args.alpha_gamma
        model.train()
        train_cm.reset()
        train_iter = tqdm(train_dataloader)
        train_losses = []
        for _, d in enumerate(train_iter):
            img, target, c = to_device(d, device)
            logits, latent = model(img)
            loss = loss_obj(torch.sigmoid(logits.squeeze()), target)

            prediction = torch.sigmoid(logits.squeeze()) > 0.5
            train_cm.update(get_stat(prediction, target))

            if args.use_mmd:
                mmd_criterion = {
                    'icl': icl,
                    'mmd': mmd_loss,
                    'mmd_lap': mmd_lap_loss
                }[args.mmd_type]
                comp_loss = mmd_criterion(args, latent, c)
                loss += args.alpha * comp_loss

            if args.use_adv:
                disc_logits, _ = disc_model(latent.detach())
                disc_loss = disc_loss_obj(disc_logits, c)
                disc_optimizer.zero_grad()
                disc_loss.backward()
                disc_optimizer.step()
                disc_logits, _ = disc_model(latent)
                disc_loss = disc_loss_obj(disc_logits, c)
                loss += -disc_loss

            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_iter.set_description("Epoch {:d} train".format(epoch))
            train_iter.set_postfix(ordered_dict={'loss': loss.item()})

        writer.add_scalar('train/loss', np.mean(train_losses), epoch)
        writer.add_scalar('train/accuracy', train_cm.get_accuracy(), epoch)
        writer.add_scalar('train/precision', train_cm.get_precision(), epoch)
        
        if (epoch + 1) % args.validation_frequecy == 0:
            # Validation loop
            model.eval()
            valid_cm.reset()
            valid_losses = []
            with torch.no_grad():
                valid_iter = tqdm(valid_dataloader)
                for _, d in enumerate(valid_iter):
                    img, target, _ = to_device(d, device)
                    logits, _ = model(img)
                    loss = loss_obj(torch.sigmoid(logits.squeeze()), target)
                    prediction = torch.sigmoid(logits.squeeze()) > 0.5
                    valid_cm.update(get_stat(prediction, target))
                    valid_losses.append(loss.item())
                    valid_iter.set_description("Epoch {:d} validation".format(epoch))
                    valid_iter.set_postfix(ordered_dict={'loss': loss.item()})

            current_acc = valid_cm.get_accuracy()
            current_loss = np.mean(valid_losses)
            writer.add_scalar('val/loss', current_loss, epoch)
            writer.add_scalar('val/accuracy', current_acc, epoch)
            writer.add_scalar('val/precision', valid_cm.get_precision(), epoch)

            if current_loss < best_valid_loss:
                print("Current validation loss {:5f} lower than best validation loss {:5f}"
                       .format(current_loss, best_valid_loss))
                best_valid_loss = current_loss
                print("Saving Model...")
                save_path = os.path.join(args.fold_dir, "best_validation_loss.pth")
                torch.save(model.state_dict(), save_path)
                print("Model saved at {:s}".format(save_path))

            if current_acc >= best_valid_acc:
                print("Current validation acc {:5f} higher than best validation acc {:5f}"
                       .format(current_acc, best_valid_acc))
                best_valid_acc = current_acc
                print("Saving Model...")
                save_path = os.path.join(args.fold_dir, "best_validation_acc.pth")
                torch.save(model.state_dict(), save_path)
                print("Model saved at {:s}".format(save_path))

        writer.flush()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", default=806, type=int)
    parser.add_argument("--logdir", default="logs/res18_16_16_32_64")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--blocks", default=[2, 2, 2, 2], type=int, nargs='+')
    parser.add_argument("--channels", default=[8, 8, 16, 32], type=int, nargs='+')
    parser.add_argument('--use_bottleneck_layers', default=False, action="store_true")
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--max_epochs", default=300, type=int)
    parser.add_argument("--validation_frequecy", default=1, type=int)
    parser.add_argument('--use_mmd', default=False, action='store_true')
    parser.add_argument('--mmd_type', default="icl", choices=['icl', 'mmd', 'mmd_lap'])
    parser.add_argument('--icl_a', type=float, default=5.0)
    parser.add_argument('--icl_b', type=float, default=0.1)

    # Additional weights to the compression term
    parser.add_argument('--alpha', type=float, default=1.0, help='Additional weight on the compression')
    parser.add_argument('--alpha_max', type=float, default=1.0, help='Max value of the regularizer')
    parser.add_argument('--alpha_gamma', type=float, default=1.5, help='Multiplicative factor for alpha')
    parser.add_argument("--lambda_1", type=float, default=0.1)
    parser.add_argument("--lambda_2", type=float, default=0.01)
    parser.add_argument('--use_adv', default=False, action='store_true')

    args = parser.parse_args()
    torch.manual_seed(args.random_seed)

    args.fold_dir = os.path.join(args.logdir, "fold_{:d}".format(args.fold))
    os.makedirs(args.fold_dir, exist_ok=True)
    main(args)