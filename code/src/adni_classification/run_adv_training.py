import argparse
import os, sys, time, shutil
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import src.adni_classification.adv_model as adv_models
import src.adni_classification.adniDataset as mydatasets

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.adni_classification.model import ResNet
from src.adni_classification.uai_model import UAIResNet
from src.average_meter import AverageMeter


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='AdniProtocolClassification')
parser.add_argument('--dataset_name', type=str, default='ADNI') # required
parser.add_argument('--model_name', type=str, default='AdniConvConti') #required
parser.add_argument('--result_path', type=str, default='result', help='output path')
parser.add_argument('--data_path', type=str, default='/mnt/AKA/mr_images/mr_machine/', 
                    help='path for data')
# Training
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('-e', '--lr_decay', type=float, default=1.,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('--seed', type=int, default=806,
                    help='Random seed to use')
parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=100, help='step size for saving trained models')

parser.add_argument('--flag_retrain', default=False, action='store_true', help='Re train')
parser.add_argument('--flag_reg', default=False, action='store_true', help='Regularizer')
parser.add_argument('--flag_plot', default=False, action='store_true', help='Plot')

# Parameters for adv models
parser.add_argument('--num_adv', type=int, default=2)
parser.add_argument('--adv_lr', type=float, default=5e-4, help='lr for the adversaries')

parser.add_argument('--adv_hidden_dim', type=int, default=64, help='hidden layers dim in adversaries')
parser.add_argument('--adv_batch_size', type=int, default=64)
parser.add_argument('--adv_num_epochs', type=int, default=150)
parser.add_argument('--adv_log_step', type=int, default=100)
parser.add_argument('--adv_use_weighted_loss', default=False, action='store_true')

# Other params
parser.add_argument('--latent_dim', type=int, default=256)

# Additional weights to the compression term
parser.add_argument('--alpha', type=float, default=1.0, help='Additional weight on the compression')

parser.add_argument('--alpha_max', type=float, default=1.0, help='Max value of the regularizer')
parser.add_argument('--alpha_gamma', type=float, default=1.5, help='Multiplicative factor for alpha')

parser.add_argument('--run_multi_adv', default=False, action='store_true', help='Runs all adversaries')

parser.add_argument('--run_type', type=str, default=None,
                    help='To create multiple runs')

parser.add_argument('--disc_lr', type=float, default=0.1, help='lr for the discriminator')
parser.add_argument('--disc_hidden_layers', type=int, default=1, help='Number of hidden layers in the adversary')
parser.add_argument('--ad_mci', default=False, action='store_true')
parser.add_argument('--run_adv_only', default=False, action='store_true')
parser.add_argument('--split_idx', type=int, default=0, help='Split index used')
parser.add_argument('--use_sigmoid', default=False, action='store_true', help='Use sigmoid')

parser.add_argument('--fold', type=int, default=0)
parser.add_argument("--blocks", default=[2, 2, 2, 2], type=int, nargs='+')
parser.add_argument("--channels", default=[16, 16, 32, 64], type=int, nargs='+')
parser.add_argument('--use_bottleneck_layers', default=False, type=bool)
parser.add_argument('--resnet_path', type=str, default=None)
parser.add_argument('--use_uai', default=False, action='store_true')

# Model parameters
args = parser.parse_args()



device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')   ### CHANGE here for cuda

torch.manual_seed(args.seed)

# Arguments
model_name=args.model_name + '-' + args.dataset_name
dataset_path=os.path.join(args.data_path, args.dataset_name)
output_path=os.path.join(args.result_path, args.experiment_name, model_name)


sample_path=os.path.join(output_path, 'samples')
log_path=os.path.join(output_path, "log.txt")
run_path = os.path.join(output_path, 'runs')

params = ['latent', args.latent_dim,
          'lr', args.lr,
          'fold', args.fold,
          'model', args.resnet_path]

params_str = '_'.join([str(x) for x in params])

if args.run_type is not None:
    params_str = args.run_type + '_' + params_str

writer_path=os.path.join(run_path, params_str)

# Models now saved in runs
model_path=os.path.join(writer_path, 'snapshots')
net_path=os.path.join(model_path, 'net.pth')
adv_path = os.path.join(model_path, 'adv.pth')


# makedir
def make_dir(dirname, rm=False):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif rm:
        print('rm and mkdir ', dirname)
        shutil.rmtree(dirname)
        os.makedirs(dirname)

if not args.run_adv_only:
    make_dir(args.result_path)
    make_dir(output_path)
    make_dir(sample_path)
    make_dir(run_path)
    make_dir(writer_path, rm=True)
    make_dir(model_path)
logf=open(log_path, 'w')
writer=SummaryWriter(comment=model_name, log_dir=writer_path)


# Logging helper functions 
def log_loss(epoch, num_epochs, step, total_step, loss, start_time):
    loss=loss.cpu().data.numpy()
    message='Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time: {:.4f}s'.format(
        epoch, num_epochs, step, total_step, loss, time.time() - start_time)
    logf.write(message + '\n')
    print(message)


### Dataloaders
trainset = mydatasets.ADNIDataset(ad_cn=True, split_filename='./data/splits/train_{:d}.data'.format(args.fold))
valset = mydatasets.ADNIDataset(ad_cn=True, split_filename='./data/splits/val_{:d}.data'.format(args.fold))


def get_prob(logits):
    logits = logits - torch.max(logits, 1)[0].unsqueeze(1)
    logits = torch.exp(logits)
    prob = logits / logits.sum(dim=1).unsqueeze(1)
    return prob[:, 1]


def train_adv_epoch(epoch, adv, opt, dataloader, writer, train=True, navib_model=None):
    # Function to train a single adversary
    if navib_model is not None:
        navib_model.eval()
    loss_logger = AverageMeter()
    correct = 0
    total = 0
    start_time = time.time()
    total_steps = len(dataloader.dataset)//args.adv_batch_size
    if train:
        adv.train()
        tag = 'train'
    else:
        adv.eval()
        tag = 'test'
    for idx, (x, _, c) in enumerate(dataloader):
        x = x.to(device)
        c = c.to(device)

        if navib_model is not None:
            with torch.no_grad():
                if args.use_uai:
                    _, _, latent, _ = navib_model(x)
                else:
                    _, latent = navib_model(x)  ## Make the resnet return latent features
                x = latent
        logits = adv(x)
        
        pred = torch.argmax(logits, 1)
        correct += torch.sum(pred == c).item()
        total += x.size(0)
        loss = F.cross_entropy(logits, c)
        loss_logger.update(loss.item())

        if idx % args.log_step == 0:
            log_loss(epoch, args.adv_num_epochs, idx, total_steps, loss, start_time)
            start_time = time.time()
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()

    accuracy = correct * 100.0 / total
    print(adv.name, tag, 'acc :', accuracy)
    writer.add_scalar(adv.name + '_loss/' + tag, loss_logger.avg, epoch)
    writer.add_scalar(adv.name + '_acc/' + tag, accuracy, epoch)
    writer.flush()


def train_adv(adv, opt, trainloader, testloader, writer, navib_model=None):
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.8)
    for epoch in range(args.adv_num_epochs):
        train_adv_epoch(epoch, adv, opt, trainloader, writer, train=True, navib_model=navib_model)
        train_adv_epoch(epoch, adv, opt, testloader, writer, train=False, navib_model=navib_model)
        lr_scheduler.step()
    print('Training done!')


def train_adversaries(trainset, valset, writer, navib_model=None):
    if navib_model is None:
        # For baseline, the adversaries train on the original data and not the latent space
        dummy_x, _, _ = trainset.__getitem__(0)
        input_dim = dummy_x.size(0)
    else:
        navib_model.eval()
        if args.use_uai:
            input_dim = args.channels[-1]
        else:
            input_dim = args.channels[-1]*2
    output_dim = 3
    hidden_dim = args.adv_hidden_dim
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.adv_batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.adv_batch_size, shuffle=False)

    i=3
    if navib_model is None:
        name = 'Baseline_Adv'
    else:
        name = navib_model.name + '_Adv'
    adv = adv_models.Adv(name + str(i), input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
                         hidden_layers=3, dropout=0.).to(device)
    opt = optim.Adam(adv.parameters(), lr=args.adv_lr)
    train_adv(adv, opt, trainloader, valloader, writer, navib_model)


def run_adversaries(trainset, valset, writer, navib_model=None):
    dummy_x, _, _ = trainset.__getitem__(0)
    if navib_model is None:
        name = 'Navib'
        path = args.resnet_path
        assert path is not None
        # Load the Resent model here
        if args.use_uai:
            model = nn.DataParallel(UAIResNet(1, args.blocks, args.channels, bottleneck=False, n_out_linear=1))
        else:
            model = nn.DataParallel(ResNet(1, args.blocks, args.channels, bottleneck=False, n_out_linear=1))
        model.load_state_dict(torch.load(path))
        model.name = name
        model.cuda()
    else:
        model = navib_model
    print('Starting to run adversary!')
    train_adversaries(trainset, valset, writer, model)


print('Running adversarial training')
run_adversaries(trainset, valset, writer, navib_model=None)