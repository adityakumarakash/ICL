import argparse
import matplotlib.pyplot as plt
import numpy as np
import os, sys, time, shutil
import torch
import torch.nn.functional as F
import torch.optim as optim

from src import dataloader as mydatasets, model as models
from src.average_meter import AverageMeter

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../..')

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='BaselineDetContiConfound')
parser.add_argument('--dataset_name', type=str, default='Adult')
parser.add_argument('--model_name', type=str, default='FC_DetEnc')
parser.add_argument('--result_path', type=str, default='result', help='output path')
parser.add_argument('--data_path', type=str, default='data/', help='path for data')

# Training
parser.add_argument('--num_epochs', type=int, default=201)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=100, help='step size for saving trained models')

parser.add_argument('--flag_retrain', default=False, action='store_true', help='Re train')
parser.add_argument('--flag_reg', default=False, action='store_true', help='Regularizer')
parser.add_argument('--flag_plot', default=False, action='store_true', help='Plot')

# Parameters for adv models
parser.add_argument('--num_adv', type=int, default=2)
parser.add_argument('--adv_lr', type=float, default=0.1, help='lr for the adversaries')
parser.add_argument('--adv_hidden_dim', type=int, default=64, help='hidden layers dim in adversaries')
parser.add_argument('--adv_batch_size', type=int, default=128)
parser.add_argument('--adv_num_epochs', type=int, default=150)
parser.add_argument('--adv_log_step', type=int, default=100)
parser.add_argument('--adv_use_weighted_loss', default=False, action='store_true')
parser.add_argument('--comp_lambda', type=float, default=1e-2)

# Other params
parser.add_argument('--latent_dim', type=int, default=30)

# Additional weights to the compression term
parser.add_argument('--alpha', type=float, default=1.0, help='Additional weight on the compression')
parser.add_argument('--alpha_max', type=float, default=1.0, help='Max value of the regularizer')
parser.add_argument('--alpha_gamma', type=float, default=1.5, help='Multiplicative factor for alpha')

parser.add_argument('--comp_type', type=str, default='adv_training', 
                    help='Choose from none, icl, adv_training')
parser.add_argument('--neighbour_threshold', type=float, default=0.05, 
                    help='The threshold to determine nearness in the nuisanace class')
parser.add_argument('--beta', type=float, default=0.05, help='The threshold for h function.')
parser.add_argument('--icl_a', type=float, default=5.0)
parser.add_argument('--icl_b', type=float, default=0.1)
parser.add_argument('--run_multi_adv', default=False, action='store_true', help='Runs all adversaries')
parser.add_argument('--run_type', type=str, default=None,
                    help='To create multiple runs')
parser.add_argument('--disc_lr', type=float, default=0.1, help='lr for the discriminator')
parser.add_argument('--disc_hidden_layers', type=int, default=1, help='Number of hidden layers in the adversary')
parser.add_argument('--adv_hidden_layers', type=int, default=3)
parser.add_argument('--gpu_ids', type=str, default="0")
parser.add_argument('--only_adv_training', default=False, action='store_true')

# Model parameters
args = parser.parse_args()


device=torch.device('cuda:{}'.format(args.gpu_ids) if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)


# Arguments
model_name=args.model_name + '-' + args.dataset_name
dataset_path=os.path.join(args.data_path, args.dataset_name)
output_path=os.path.join(args.result_path, args.experiment_name, model_name)

sample_path=os.path.join(output_path, 'samples')
log_path=os.path.join(output_path, "log.txt")
run_path = os.path.join(output_path, 'runs')

params = ['latent', args.latent_dim, 
          'alpha', args.alpha,
          'alpha_max', args.alpha_max,
          'alpha_gamma', args.alpha_gamma,
          'neighnour_threshold', args.neighbour_threshold,
          'type', args.comp_type,
          'seed', args.seed]

if args.comp_type == 'icl':
    add_params = ['icl_a', args.icl_a, 'icl_b', args.icl_b]
    params.extend(add_params)

if args.comp_type == 'adv_training':
    add_params = ['disc_lr', args.disc_lr,
                  'disc_layers', args.disc_hidden_layers]
    params.extend(add_params)
    
params_str = '_'.join([str(x) for x in params])

if args.run_type is not None:
    params_str = args.run_type + '_' + params_str
if args.adv_use_weighted_loss:
    params_str += '_advw'

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


if not args.only_adv_training:
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


print(args.dataset_name)
if args.dataset_name == 'Adult':
    args.data_path = os.path.join(args.data_path, 'adult_age_proc.z')
elif args.dataset_name == 'German':
    args.data_path = os.path.join(args.data_path, 'german_proc.z')
else:
    raise NotImplementedError

trainset = mydatasets.BaselineDataset(args.data_path, split='train', continuous_confound=True)
valset = mydatasets.BaselineDataset(args.data_path, split='val', continuous_confound=True)
testset = mydatasets.BaselineDataset(args.data_path, split='test', continuous_confound=True)


def get_prob(logits):
    logits = logits - torch.max(logits, 1)[0].unsqueeze(1)
    logits = torch.exp(logits)
    prob = logits / logits.sum(dim=1).unsqueeze(1)
    return prob[:, 1]


def train_adv_epoch(epoch, adv, opt, dataloader, writer, tag='train', navib_model=None):
    # Function to train a single adversary
    loss_logger = AverageMeter()
    start_time = time.time()
    total_steps = len(dataloader.dataset)//args.adv_batch_size
    train = tag == 'train'
    if train:
        adv.train()
    else:
        adv.eval()
    for idx, (x, _, c) in enumerate(dataloader):
        x = x.to(device)
        c = c.to(device)
        if navib_model is not None:
            if args.comp_type == 'uai':
                _, _, latent, _ = navib_model(x)
            else:
                _, latent = navib_model(x)
            x = latent
        pred = adv(x)
        loss = F.mse_loss(pred, c)
        loss_logger.update(loss.item())
        if idx % args.log_step == 0:
            log_loss(epoch, args.adv_num_epochs, idx, total_steps, loss, start_time)
            start_time = time.time()
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
    print(adv.name, tag, 'mse_loss :', loss_logger.avg*1e4)
    writer.add_scalar(adv.name + '_loss/' + tag, loss_logger.avg*1e4, epoch)


def train_adv(adv, opt, trainloader, valloader, testloader, writer, navib_model=None):
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.65)
    for epoch in range(args.adv_num_epochs):
        train_adv_epoch(epoch, adv, opt, trainloader, writer, tag='train', navib_model=navib_model)
        train_adv_epoch(epoch, adv, opt, valloader, writer, tag='val', navib_model=navib_model)
        train_adv_epoch(epoch, adv, opt, testloader, writer, tag='test', navib_model=navib_model)
        lr_scheduler.step()
    print('Training done!')


def train_adversaries(trainset, valset, testset, writer, navib_model=None):
    if navib_model is None:
        # For baseline, the adversaries train on the original data and not the latent space
        dummy_x, _, _ = trainset.__getitem__(0)
        input_dim = dummy_x.size(0)
    else:
        navib_model.eval()
        input_dim = args.latent_dim# * 2
    output_dim = 1
    hidden_dim = args.adv_hidden_dim
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.adv_batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.adv_batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.adv_batch_size, shuffle=False)
    
    adv_hidden_layers = 3
    if navib_model is None:
        name = 'Baseline_Adv'
    else:
        name = navib_model.name + '_Adv'
    adv = models.Adv(name + str(adv_hidden_layers), input_dim=input_dim, output_dim=output_dim,
                     hidden_dim=hidden_dim, hidden_layers=adv_hidden_layers).to(device)
    opt = optim.Adam(adv.parameters(), lr=args.adv_lr)
    train_adv(adv, opt, trainloader, valloader, testloader, writer, navib_model)


######################### Navib Model ################################
def icl_loss(z, confounds):
    # Returns the icl loss with hybrid kernel
    diff_z = (z.unsqueeze(1) - z.unsqueeze(0)).norm(dim=-1)
    pull_cost = diff_z.pow(2)
    push_cost = torch.exp(args.icl_a - args.icl_b * diff_z)
    
    diff_c = torch.abs(confounds.unsqueeze(1) - confounds.unsqueeze(0))
    threshold = args.neighbour_threshold
    loss = pull_cost[diff_c > threshold].sum()
    loss += push_cost[diff_c <= threshold].sum()
    n = z.size(0)*1.0
    loss /= n*n  # Mean cost
    return loss


def train_disc(epoch, adv, opt, trainloader, writer, navib_model):
    assert(navib_model is not None)
    navib_model.eval()
    train_adv_epoch(epoch, adv, opt, trainloader, writer, tag='train', navib_model=navib_model)
    print('Disc epoch!')


def train_disentangler(epoch, dis, opt, trainloader, writer, navib_model):
    navib_model.eval()
    for idx, (x, y, c) in enumerate(trainloader):
        x = x.to(device)
        _, _, e1, e2 = navib_model(x)
        e1 = e1.detach()
        e2 = e2.detach()
        e1_pred, e2_pred = dis(e1, e2)
        loss = F.mse_loss(e1_pred, e1) + F.mse_loss(e2_pred, e2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print('Disentangler epoch!')


def navib_epoch(epoch, model, opt, dataloader, writer, tag='train', disc=None, disc_opt=None):
    loss_logger = AverageMeter() # Total loss logger
    pred_loss_logger = AverageMeter()
    comp_loss_logger = AverageMeter()
    latent_logger = AverageMeter()
    train = tag == 'train'
    if train:
        if args.comp_type == 'adv_training':
            train_disc(epoch, disc, disc_opt, dataloader, writer, model)
            disc.eval()
        model.train()
    else:
        model.eval()
    total_steps = len(dataloader.dataset)//args.batch_size
    
    y_correct = 0
    y_total = 0
    y_true_pos = 0
    y_pos = 0
    
    start_time = time.time()

    for idx, (x, y, c) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)
        if args.comp_type == 'uai':
            recons, pred_logits, latent, latent2 = model(x)
            recons_loss = F.mse_loss(recons, x)  # , reduction='sum')
            pred_loss = F.cross_entropy(pred_logits, y)  # , reduction='sum')
            if train:
                e1_pred, e2_pred = disc(latent, latent2)
                rand_target1 = torch.rand(size=e1_pred.size()).to(device)
                rand_target2 = torch.rand(size=e2_pred.size()).to(device)
                comp_loss = F.mse_loss(e1_pred, rand_target1) + F.mse_loss(e2_pred, rand_target2)
            else:
                comp_loss = torch.tensor(0).to(device)
            loss = args.comp_lambda * recons_loss + pred_loss + args.alpha * comp_loss
        else:
            pred_logits, latent = model(x)
            pred_loss = F.cross_entropy(pred_logits, y)

            if args.comp_type == 'none':
                comp_loss = torch.tensor(0).to(device)
            elif args.comp_type == 'icl':
                comp_loss = icl_loss(latent, c)
            elif args.comp_type == 'adv_training':
                if train:
                    logits = disc(latent)
                    comp_loss = -1 * F.mse_loss(logits, c)
                else:
                    comp_loss = torch.tensor(0).to(device)
            else:
                raise NotImplementedError
            loss = pred_loss + args.alpha * comp_loss

        pred = torch.argmax(pred_logits, 1)
        y_correct += torch.sum(pred == y)
        y_total += x.size(0)
        y_pos += torch.sum(y)
        y_true_pos += torch.sum(y[pred == 1])
        
        # Log the losses
        loss_logger.update(loss.item())
        pred_loss_logger.update(pred_loss.item())
        comp_loss_logger.update(comp_loss.item())
        latent_logger.update(latent.norm(dim=-1).mean())
        
        if idx % args.log_step == 0:
            log_loss(epoch, args.num_epochs, idx, total_steps, loss, start_time)
            start_time = time.time()
        
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    model_name = 'navib_'
    accuracy = y_correct * 100.0 / y_total
    precision = y_true_pos * 100.0 / y_pos
    print(tag, 'accuracy:', accuracy.item())
    writer.add_scalar(model_name + 'acc/' + tag, accuracy, epoch)
    writer.add_scalar(model_name + 'pred_loss/' + tag, pred_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'comp_loss/' + tag, comp_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'mu/' + tag, latent_logger.avg, epoch)
    writer.add_scalar(model_name + 'loss/' + tag, loss_logger.avg, epoch)
    

def train_navib(model, opt, trainloader, valloader, testloader, writer):
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.65)
    if args.comp_type == 'adv_training':
        disc = models.Adv('Disc', input_dim=args.latent_dim, output_dim=1,
                          hidden_dim=args.adv_hidden_dim, hidden_layers=args.disc_hidden_layers).to(device)
        disc_opt = optim.Adam(disc.parameters(), lr=args.disc_lr)
    elif args.comp_type == 'uai':
        disc = models.UAIDisentangler(latent_dim=args.latent_dim).to(device)
        disc_opt = optim.Adam(disc.parameters(), lr=args.disc_lr)
    for epoch in range(1, args.num_epochs + 1):
        if args.comp_type == 'adv_training' or args.comp_type == 'uai':
            navib_epoch(epoch, model, opt, trainloader, writer, tag='train', disc=disc, disc_opt=disc_opt)
        else:
            navib_epoch(epoch, model, opt, trainloader, writer, tag='train')
        navib_epoch(epoch, model, opt, valloader, writer, tag='val')
        navib_epoch(epoch, model, opt, testloader, writer, tag='test')
        if epoch % args.save_step == 0:
            name = 'Navib_' + str(epoch)
            path = os.path.join(model_path, name + '.pth')
            model.name = name
            torch.save(model.state_dict(), path)
        lr_scheduler.step()
        if args.alpha < args.alpha_max:
            args.alpha *= args.alpha_gamma
    name = 'Navib'
    model.name = name
    path = os.path.join(model_path, name + '.pth')
    torch.save(model.state_dict(), path)
    

def navib(trainset, valset, testset, writer):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    dummy_x, _, _ = trainset.__getitem__(0)
    input_dim = dummy_x.size(0)
    if args.comp_type == 'uai':
        model = models.UAIModel(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    else:
        model = models.FC_DetEnc(input_dim=input_dim, latent_dim=args.latent_dim,
                                 hidden_dim=64, output_dim=2).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    train_navib(model, opt, trainloader, valloader, testloader, writer)


if not args.only_adv_training:
    navib(trainset, valset, testset, writer)


#############  Adversary after trained model  ####################
def run_adversaries(trainset, valset, testset, writer):
    dummy_x, _, _ = trainset.__getitem__(0)
    input_dim = dummy_x.size(0)
    name = 'Navib'
    path = os.path.join(model_path, name + '.pth')
    if args.comp_type == 'uai':
        model = models.UAIModel(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    else:
        model = models.FC_DetEnc(input_dim=input_dim, latent_dim=args.latent_dim,
                             hidden_dim=64, output_dim=2).to(device)
    model.load_state_dict(torch.load(path))
    model.name = name
    train_adversaries(trainset, valset, testset, writer, model)


print('Running Adversarial training now')
run_adversaries(trainset, valset, testset, writer)
print('Adversarial training done')