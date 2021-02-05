import argparse
from src import dataloader as mydatasets, model as models
import os, sys, time, shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.average_meter import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
sys.path.append('../..')


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='BaselineComparison')
parser.add_argument('--dataset_name', type=str, default='Adult')
parser.add_argument('--model_name', type=str, default='FC')
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
parser.add_argument('--adv_num_epochs', type=int, default=250)
parser.add_argument('--adv_log_step', type=int, default=100)
parser.add_argument('--adv_use_weighted_loss', default=False, action='store_true')

# Other params
parser.add_argument('--latent_dim', type=int, default=30)
parser.add_argument('--comp_lambda', type=float, default=1e-2)
parser.add_argument('--beta', type=float, default=1e-2)

# Additional weights to the compression term
parser.add_argument('--alpha', type=float, default=0.1, help='Additional weight on the compression')
parser.add_argument('--alpha_max', type=float, default=10.0, help='Max value of the regularizer') # 100
parser.add_argument('--alpha_gamma', type=float, default=1.5, help='Multiplicative factor for alpha')  # 1.5

parser.add_argument('--comp_type', type=str, default='none', 
                    help='Choose from kl, ot_pairwise, none,' + 'mmd_loss,' + 'icl, mmd_lap')

parser.add_argument('--run_type', type=str, default=None,
                    help='To create multiple runs')
parser.add_argument('--add_prior', default=False, action='store_true', 
                    help='Add the gaussian prior term like VAE')

parser.add_argument('--use_weighted_loss', default=False, action='store_true')
parser.add_argument('--run_multi_adv', default=False, action='store_true', 
                    help='Runs multiple adversaries for this experiment.')

parser.add_argument('--icl_a', type=float, default=1.0, help='Argument a for ICL')
parser.add_argument('--icl_b', type=float, default=1.0, help='Argument b for ICL')
parser.add_argument('--mmd_lap_p', type=float, default=1.0, 
                    help='Argument for mmd laplacian')
parser.add_argument('--disc_lr', type=float, default=0.1, help='lr for the discriminator')
parser.add_argument('--adv_hidden_layers', type=int, default=3)
parser.add_argument('--gpu_ids', type=str, default="0")


# Model parameters
args = parser.parse_args()

device=torch.device('cuda:{}'.format(args.gpu_ids) if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)


# Arguments
model_name=args.model_name + '-' + args.dataset_name
dataset_path=os.path.join(args.data_path, args.dataset_name)
output_path=os.path.join(args.result_path, args.experiment_name, model_name)

# saved checkpoint
sample_path=os.path.join(output_path, 'samples')
log_path=os.path.join(output_path, "log.txt")
run_path = os.path.join(output_path, 'runs')


params = ['latent', args.latent_dim,
          'beta', args.beta,
          'comp_lambda', args.comp_lambda,
          'alpha', args.alpha,
          'alpha_max', args.alpha_max,
          'alpha_gamma', args.alpha_gamma,
          'type', args.comp_type,
          'seed', args.seed]

if args.comp_type == 'icl':
    add_params = ['icl_a', args.icl_a, 
                  'icl_b', args.icl_b]
    params.extend(add_params)
    
if args.comp_type == 'mmd_lap':
    add_params = ['lap_p', args.mmd_lap_p]
    params.extend(add_params)
    
if args.comp_type == 'adv_training':
    add_params = ['disc_lr', args.disc_lr]
    params.extend(add_params)

params_str = '_'.join([str(x) for x in params])

if args.run_type is not None:
    params_str = args.run_type + '_' + params_str
if args.add_prior:
    params_str += '_klprior'
if args.adv_use_weighted_loss:
    params_str += '_advw'

writer_path=os.path.join(run_path, params_str)   # Helps to identify the hyperparamters associated with the runs.

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


make_dir(args.result_path)
make_dir(output_path)
make_dir(sample_path)
make_dir(run_path)
logf=open(log_path, 'w')
make_dir(writer_path, rm=True)
make_dir(model_path)
writer=SummaryWriter(comment=model_name, log_dir=writer_path)


# Logging helper function
def log_loss(epoch, num_epochs, step, total_step, loss, start_time):
    loss=loss.cpu().data.numpy()
    message='Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time: {:.4f}s'.format(
        epoch, num_epochs, step, total_step, loss, time.time() - start_time)
    logf.write(message + '\n')
    print(message)


#### Dataloaders
if args.dataset_name == 'Adult':
    args.data_path = os.path.join(args.data_path, 'adult_proc.z')
elif args.dataset_name == 'German':
    args.data_path = os.path.join(args.data_path, 'german_proc.z')
else:
    raise NotImplementedError

trainset = mydatasets.BaselineDataset(args.data_path, split='train')
valset = mydatasets.BaselineDataset(args.data_path, split='val')
testset = mydatasets.BaselineDataset(args.data_path, split='test')


# In this we train the model on the actual dataset and not the latent space.
def get_prob(logits):
    logits = logits - torch.max(logits, 1)[0].unsqueeze(1)
    logits = torch.exp(logits)
    prob = logits / logits.sum(dim=1).unsqueeze(1)
    return prob[:, 1]


def train_adv_epoch(epoch, adv, opt, dataloader, writer, tag='train', navib_model=None):
    # Function to train a single adversary
    loss_logger = AverageMeter()
    correct = 0
    total = 0
    pos = 0
    true_pos = 0
    start_time = time.time()
    total_steps = len(dataloader.dataset)//args.adv_batch_size
    if args.adv_use_weighted_loss:
        weights = dataloader.dataset.get_confound_weights().to(device)
    train = tag == 'train'
    if train:
        adv.train()
    else:
        adv.eval()
    y_true = np.array([])
    y_score = np.array([])
    for idx, (x, _, c) in enumerate(dataloader):
        x = x.to(device)
        c = c.to(device)
        if navib_model is not None:
            if args.comp_type == 'uai':
                _, _, mu, _ = navib_model(x)
            else:
                _, _, mu, _ = navib_model(x, c.float().unsqueeze(1))
            x = mu
        logits = adv(x)
        
        # For computing the auc roc
        y_true = np.concatenate((y_true, c.cpu().numpy()))
        y_score = np.concatenate((y_score, get_prob(logits).detach().cpu().numpy()))
        
        pred = torch.argmax(logits, 1)
        correct += torch.sum(pred == c)
        total += x.size(0)
        pos += torch.sum(c)
        true_pos += torch.sum(c[pred == 1])
        if args.adv_use_weighted_loss:
            loss = F.cross_entropy(logits, c, weights)
        else:
            loss = F.cross_entropy(logits, c)
        loss_logger.update(loss.item())
        if idx % args.log_step == 0:
            log_loss(epoch, args.adv_num_epochs, idx, total_steps, loss, start_time)
            start_time = time.time()
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
    roc_auc = roc_auc_score(y_true, y_score)
    accuracy = correct.item() * 100.0 / total
    precision = true_pos * 100.0 / pos
    print(adv.name, tag, 'acc :', accuracy)
    print(adv.name, tag, 'precision :', precision.item())
    print(adv.name, tag, 'roc_auc :', roc_auc)
    writer.add_scalar(adv.name + '_loss/' + tag, loss_logger.avg, epoch)
    writer.add_scalar(adv.name + '_acc/' + tag, accuracy, epoch)
    writer.add_scalar(adv.name + '_precision/' + tag, precision.item(), epoch)
    writer.add_scalar(adv.name + '_roc_auc/' + tag, roc_auc, epoch)


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
        input_dim = args.latent_dim
    output_dim = 2
    hidden_dim = args.adv_hidden_dim
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.adv_batch_size, shuffle=True, drop_last=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.adv_batch_size, shuffle=False, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.adv_batch_size, shuffle=False, drop_last=True)
    
    adv_hidden_layers = args.adv_hidden_layers
    if navib_model is None:
        name = 'Baseline_Adv'
    else:
        name = navib_model.name + '_Adv'
    adv = models.Adv(name + str(adv_hidden_layers), input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
                     hidden_layers=adv_hidden_layers).to(device)
    opt = optim.Adam(adv.parameters(), lr=args.adv_lr)
    train_adv(adv, opt, trainloader, valloader, testloader, writer, navib_model)


######################### Navib Model ################################
def prior_kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def gaussian_entropy(mu, logvar):
    loss = 0.5 * torch.sum(logvar + 1e-8, -1)
    B = mu.size(0)
    return (1.0*(B-1)/B) * loss.mean()


def kl_loss(mu, logvar):
    # Pairwise computation of the KL for approximation of the KL term
    sigma = torch.exp(0.5 * logvar)
    sigma_sq = torch.pow(sigma, 2) + 1e-8
    sigma_sq_inv = 1.0 / sigma_sq
    
    # First term
    first_term = torch.mm(sigma_sq, sigma_sq_inv.transpose(0, 1))
    
    r = torch.mm(mu*mu, sigma_sq_inv.transpose(0, 1))
    r2 = mu*mu*sigma_sq_inv
    r2 = torch.sum(r2, 1)
    
    second_term = 2*torch.mm(mu, torch.transpose(mu*sigma_sq_inv, 0, 1))
    second_term = r - second_term + r2.unsqueeze(1).transpose(0, 1)
    
    r = torch.sum(torch.log(sigma_sq), 1)
    r = torch.reshape(r, [-1, 1])
    third_term = r - r.transpose(0, 1)
    
    loss = 0.5 * (first_term + second_term + third_term)
    loss = loss.mean()
    return loss


def ot_pairwise_loss(mu, logvar):
    # Here we approximate the wasserstein loss using the pairwise wassersteins
    # This uses the closed form for the normals 
    sigma = torch.exp(0.5 * logvar)
    diff_sigma = sigma.unsqueeze(0) - sigma.unsqueeze(1)
    diff_mu = mu.unsqueeze(0) - mu.unsqueeze(1)
    loss = diff_mu.norm(dim=-1).pow(2).mean()
    loss += diff_sigma.norm(dim=-1).pow(2).mean()
    return loss


def icl(mu, labels, logvar):
    diff = mu.unsqueeze(0) - mu.unsqueeze(1)
    diff = diff.norm(dim = -1)
    sq_loss = diff.pow(2)
    lap_kernel = torch.exp(args.icl_a - args.icl_b * diff)

    s = torch.zeros_like(labels).float()
    for i in range(2):
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            continue
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)
    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    loss = (sq_loss * s_prod)[c_diff == True].sum()
    loss += 1 * (lap_kernel * s_prod)[c_diff == False].sum()

    sigma = torch.exp(0.5 * logvar)
    diff_sigma = sigma.unsqueeze(0) - sigma.unsqueeze(1)
    loss += diff_sigma.pow(2).sum(-1).mean()
    return loss


def mmd_loss(mu, labels, logvar):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.pow(2).sum(-1)

    s = torch.zeros_like(labels).float()
    for i in range(2):
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            # Otherwise goes negative
            return torch.tensor(0.0).to(device)
            continue
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)

    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    diff = diff * s_prod
    loss = diff[c_diff == True].sum()
    loss += -1*diff[c_diff == False].sum()
    
    sigma = torch.exp(0.5 * logvar)
    diff_sigma = sigma.unsqueeze(0) - sigma.unsqueeze(1)
    loss += diff_sigma.pow(2).sum(-1).mean()
    
    return loss


def mmd_lap_loss(mu, labels, logvar):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.norm(dim = -1)
    lap_kernel = torch.exp(-diff/args.mmd_lap_p)

    s = torch.zeros_like(labels).float()
    for i in range(2):
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            continue
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)

    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    lap_kernel = lap_kernel * s_prod
    loss = -lap_kernel[c_diff == True].sum()
    loss += 1*lap_kernel[c_diff == False].sum()
    
    sigma = torch.exp(0.5 * logvar)
    diff_sigma = sigma.unsqueeze(0) - sigma.unsqueeze(1)
    loss += diff_sigma.pow(2).sum(-1).mean()
    
    return loss


def train_disc(epoch, adv, opt, trainloader, writer, navib_model):
    assert navib_model is not None
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
    loss_logger = AverageMeter()
    recons_loss_logger = AverageMeter()
    pred_loss_logger = AverageMeter()
    comp_loss_logger = AverageMeter()
    mu_logger = AverageMeter()
    sigma_logger = AverageMeter()
    prior_loss_logger = AverageMeter()
    train = tag == 'train'
    if train:
        if args.comp_type == 'adv_training':
            train_disc(epoch, disc, disc_opt, dataloader, writer, model)
            disc.eval()
        elif args.comp_type == 'uai':
            k = 5
            for i in range(k):
                train_disentangler(epoch, disc, disc_opt, dataloader, writer, model)
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
            recons_loss = F.mse_loss(recons, x)
            pred_loss = F.cross_entropy(pred_logits, y)
            if train:
                e1_pred, e2_pred = disc(latent, latent2)
                rand_target1 = torch.rand(size=e1_pred.size()).to(device)
                rand_target2 = torch.rand(size=e2_pred.size()).to(device)
                comp_loss = F.mse_loss(e1_pred, rand_target1) + F.mse_loss(e2_pred, rand_target2)
            else:
                comp_loss = torch.tensor(0).to(device)
            loss = args.comp_lambda * recons_loss + pred_loss + args.alpha * comp_loss
        else:
            recons, pred_logits, mu, logvar = model(x, c.unsqueeze(1))
            sigma = torch.exp(0.5 * logvar)
            prior_loss = prior_kl_loss(mu, logvar)

            recons_loss = F.mse_loss(recons, x)
            pred_loss = F.cross_entropy(pred_logits, y)

            if args.comp_type == 'none':
                comp_loss = torch.tensor(0).to(device)
            elif args.comp_type == 'kl':
                comp_loss = kl_loss(mu, logvar)
            elif args.comp_type == 'ot_pairwise':
                comp_loss = ot_pairwise_loss(mu, logvar)
            elif args.comp_type == 'mmd_loss':
                comp_loss = mmd_loss(mu, c, logvar)
            elif args.comp_type == 'icl':
                comp_loss = icl(mu, c, logvar)
            elif args.comp_type == 'mmd_lap':
                comp_loss = mmd_lap_loss(mu, c, logvar)
            elif args.comp_type == 'adv_training':
                if train:
                    logits = disc(mu)
                    comp_loss = -1 * F.cross_entropy(logits, c)
                else:
                    comp_loss = torch.tensor(0).to(device)
            else:
                raise NotImplementedError

            loss = args.comp_lambda * recons_loss + pred_loss + args.alpha * (args.comp_lambda + args.beta) * comp_loss

            if args.add_prior:
                loss += args.beta * prior_loss
                prior_loss_logger.update(prior_loss.item())

            mu_logger.update(mu.norm(dim=-1).mean())
            sigma_logger.update(sigma.norm(dim=-1).mean().item())

        # Log the losses
        loss_logger.update(loss.item())
        recons_loss_logger.update(recons_loss.item())
        pred_loss_logger.update(pred_loss.item())
        comp_loss_logger.update(comp_loss.item())

        pred = torch.argmax(pred_logits, 1)
        y_correct += torch.sum(pred == y)
        y_total += x.size(0)
        y_pos += torch.sum(y)
        y_true_pos += torch.sum(y[pred == 1])

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

    writer.add_scalar(model_name + 'recons_loss/' + tag, recons_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'pred_loss/' + tag, pred_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'comp_loss/' + tag, comp_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'mu/' + tag, mu_logger.avg, epoch)
    writer.add_scalar(model_name + 'sigma/' + tag, sigma_logger.avg, epoch)
    writer.add_scalar(model_name + 'loss/' + tag, loss_logger.avg, epoch)


def train_navib(model, opt, trainloader, valloader, testloader, writer):
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.65)
    if args.comp_type == 'adv_training':
        disc = models.Adv('Disc', input_dim=args.latent_dim, output_dim=2,
                          hidden_dim=args.adv_hidden_dim, hidden_layers=3).to(device)
        disc_opt = optim.Adam(disc.parameters(), lr=args.disc_lr)
    elif args.comp_type == 'uai':
        disc = models.UAIDisentangler(latent_dim=args.latent_dim).to(device)
        disc_opt = optim.Adam(disc.parameters(), lr=args.disc_lr)
    for epoch in range(1, args.num_epochs + 1):
        if args.comp_type == 'adv_training' or args.comp_type == 'uai':
            navib_epoch(epoch, model, opt, trainloader, writer, tag='train',
                        disc=disc, disc_opt=disc_opt)
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


def navib(args, trainset, valset, testset, writer):
    if args.dataset_name == 'German' and (args.comp_type == 'adv_training' or args.comp_type == 'uai'):
        drop_last = True
    else:
        drop_last = False
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              drop_last=drop_last)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    dummy_x, _, _ = trainset.__getitem__(0)
    input_dim = dummy_x.size(0)

    if args.comp_type == 'uai':
        model = models.UAIModel(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    else:
        model = models.BaselineVAE(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=1).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    train_navib(model, opt, trainloader, valloader, testloader, writer)


navib(args, trainset, valset, testset, writer)


####################### Adversary after trained model #######################

def run_adversaries(trainset, valset, testset, writer):
    dummy_x, _, _ = trainset.__getitem__(0)
    input_dim = dummy_x.size(0)
    name = 'Navib'
    path = os.path.join(model_path, name + '.pth')
    if args.comp_type == 'uai':
        model = models.UAIModel(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    else:
        model = models.BaselineVAE(input_dim=input_dim, latent_dim=args.latent_dim, feature_dim=1).to(device)
    model.load_state_dict(torch.load(path))
    model.name = name
    train_adversaries(trainset, valset, testset, writer, model)


print('Running Adversarial training now')
run_adversaries(trainset, valset, testset, writer)
print('Adversarial training done')