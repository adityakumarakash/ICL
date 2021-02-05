import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.adni_classification.utils import ConfusionMatrix
from src.adni_classification.adniDataset import ADNIDataset
from src.adni_classification.uai_model import UAIResNet, UAIDisentangler

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


def to_device(list_of_tensors, device):
    return [t.to(device) for t in list_of_tensors]


def train_disentangler(dis, opt, trainloader, resnet_model):
    resnet_model.eval()
    for _, d in enumerate(trainloader):
        x, target, c = to_device(d, device)
        x = x.to(device)
        _, _, e1, e2 = resnet_model(x)
        e1 = e1.detach()
        e2 = e2.detach()
        e1_pred, e2_pred = dis(e1, e2)
        loss = F.mse_loss(e1_pred, e1) + F.mse_loss(e2_pred, e2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print('Disentangler epoch!')


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

    model = UAIResNet(1, args.blocks, args.channels, bottleneck=args.use_bottleneck_layers,
                      n_out_linear=1, dropout=0.5)
    print('Model:', model)
    model = nn.DataParallel(model).to(device)

    dis = nn.DataParallel(UAIDisentangler(64)).to(device)
    dis_optimizer = torch.optim.Adam(dis.parameters(), lr=args.lr)
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

        if args.use_uai:
            k = 5
            model.eval()
            for i in range(k):
                train_disentangler(dis, dis_optimizer, train_dataloader, model)

        model.train()
        train_cm.reset()
        train_iter = tqdm(train_dataloader)
        train_losses = []

        for _, d in enumerate(train_iter):
            img, target, c = to_device(d, device)
            if args.use_uai:
                logits, recons, latent1, latent2 = model(img)
            else:
                logits, latent = model(img)
            loss = loss_obj(torch.sigmoid(logits.squeeze()), target)

            prediction = torch.sigmoid(logits.squeeze()) > 0.5
            train_cm.update(get_stat(prediction, target))

            if args.use_uai:
                recons_loss = F.mse_loss(recons, img)
                random_target1 = torch.rand(size=latent1.size()).to(device)
                random_target2 = torch.rand(size=latent1.size()).to(device)
                dis_loss = F.mse_loss(latent1, random_target1) + F.mse_loss(latent2, random_target2)
                loss += recons_loss * 0.1 + args.alpha * dis_loss

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
                    if args.use_uai:
                        logits, _, _, _ = model(img)
                    else:
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

    # Additional weights to the compression term
    parser.add_argument('--alpha', type=float, default=1.0, help='Additional weight on the compression')
    parser.add_argument('--alpha_max', type=float, default=1.0, help='Max value of the regularizer')
    parser.add_argument('--alpha_gamma', type=float, default=1.5, help='Multiplicative factor for alpha')
    parser.add_argument("--lambda_1", type=float, default=0.1)
    parser.add_argument("--lambda_2", type=float, default=0.01)
    parser.add_argument('--use_uai', default=False, action='store_true')

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    args.fold_dir = os.path.join(args.logdir, "fold_{:d}".format(args.fold))
    os.makedirs(args.fold_dir, exist_ok=True)
    main(args)