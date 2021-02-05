import argparse
import os, sys, time, shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import datasets as vision_datasets
from torch.utils.data.sampler import SubsetRandomSampler

from src import dataloader as mydatasets, model as models
from src.average_meter import AverageMeter


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='DetRotatedMNSIT')
parser.add_argument('--dataset_name', type=str, default='RotMNIST') # required
parser.add_argument('--model_name', type=str, default='FC') #required
parser.add_argument('--result_path', type=str, default='result', help='output path')
parser.add_argument('--data_path', type=str, default='data/', help='path for data')

# Training
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('--log_step', type=int, default=200,
                    help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=100,
                    help='step size for saving trained models')

parser.add_argument('--flag_retrain', default=False, action='store_true', help='Re train')
parser.add_argument('--flag_reg', default=False, action='store_true', help='Regularizer')
parser.add_argument('--flag_plot', default=False, action='store_true', help='Plot')

# Parameters for adv models
parser.add_argument('--num_adv', type=int, default=2)
parser.add_argument('--adv_lr', type=float, default=1e-1, help='lr for the adversaries')

parser.add_argument('--adv_hidden_dim', type=int, default=64, help='hidden layers dim in adversaries')
parser.add_argument('--adv_batch_size', type=int, default=128)
parser.add_argument('--adv_num_epochs', type=int, default=200)
parser.add_argument('--adv_log_step', type=int, default=100)
parser.add_argument('--adv_use_weighted_loss', default=False, action='store_true')

# Other params
parser.add_argument('--latent_dim', type=int, default=100)

# Additional weights to the compression term
parser.add_argument('--alpha', type=float, default=1.0, help='Additional weight on the compression')

parser.add_argument('--alpha_max', type=float, default=1.0, help='Max value of the regularizer')
parser.add_argument('--alpha_gamma', type=float, default=1.5, help='Multiplicative factor for alpha')

parser.add_argument('--comp_type', type=str, default='adv_training', 
                    help='Choose from ot_pairwise, none, inv_contrastive, icl, mmd_loss, mmd_lap,' + \
                         'mmd_gkernel, adv_training')

parser.add_argument('--run_type', type=str, default=None,
                    help='To create multiple runs')

parser.add_argument('--use_weighted_loss', default=False, action='store_true')

parser.add_argument('--icl_a', type=float, default=5.0, help='Argument a for ICL')
parser.add_argument('--icl_b', type=float, default=0.1, help='Argument b for ICL')
parser.add_argument('--mmd_lap_p', type=float, default=2.0, 
                    help='Argument for mmd laplacian')

parser.add_argument('--disc_lr', type=float, default=0.001, help='lr for the discriminator')
parser.add_argument('--disc_hidden_layers', type=int, default=0, help='Number of hidden layers in the adversary')
parser.add_argument('--train_adv_only', default=False, action='store_true')

parser.add_argument('--base_p', default=8, type=int)
parser.add_argument('--only_adv_training', default=False, action='store_true')
parser.add_argument('--beta', default=1.0, type=float)
parser.add_argument('--latent_dim2', default=20, type=int)
parser.add_argument('--gpu_ids', type=str, default='0')
parser.add_argument('--pred_lambda', default=100.0, type=float)


# Model parameters
args = parser.parse_args()
os.environ['CUDA_AVAILABLE_DEVICES'] = args.gpu_ids
device=torch.device('cuda:{}'.format(args.gpu_ids)
                    if torch.cuda.is_available() else 'cpu')
torch.manual_seed(args.seed)

# Arguments
model_name=args.model_name + '-' + args.dataset_name
dataset_path=os.path.join(args.data_path, args.dataset_name)
output_path=os.path.join(args.result_path, args.experiment_name, model_name)

# saved checkpoint
sample_path=os.path.join(output_path, 'samples')
log_path=os.path.join(output_path, "log.txt")
run_path = os.path.join(output_path, 'runs')

params = ['type', args.comp_type,
          'latent', args.latent_dim,
          'alpha', args.alpha,
          'alpha_max', args.alpha_max,
          'alpha_gamma', args.alpha_gamma,
          'beta', args.beta,
          'pred_lambda', args.pred_lambda,
          'seed', args.seed]

if args.dataset_name == 'RotMNISTControl':
    params.extend(['base_p', args.base_p])

if args.comp_type == 'icl' or args.comp_type == 'mmd_gkernel':
    add_params = ['icl_a', args.icl_a, 
                  'icl_b', args.icl_b]
    params.extend(add_params)
    
if args.comp_type == 'mmd_lap':
    add_params = ['lap_p', args.mmd_lap_p]
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
logf=open(os.path.join(output_path, "log.txt"), 'w')
writer=SummaryWriter(comment=model_name, log_dir=writer_path)


# Logging helper functions 
def log_loss(epoch, num_epochs, step, total_step, loss, start_time):
    loss=loss.cpu().data.numpy()
    message='Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time: {:.4f}s'.format(
        epoch, num_epochs, step, total_step, loss, time.time() - start_time)
    print(message)


# Dummy code to download the MNIST datasets.
_ = vision_datasets.MNIST(root='./data', train=True, download=True, transform=None)
_ = vision_datasets.MNIST(root='./data', train=False, download=True, transform=None)

unit_angle = 22.5
transform = transforms.Compose([transforms.ToTensor()])
if args.dataset_name == 'RandomRotMNIST':
    trainset = mydatasets.RandomRotMNIST(root='./data', unit_angle=unit_angle,
                                   train=True, download=False, transform=transform)
    valset = mydatasets.RandomRotMNIST(root='./data', unit_angle=unit_angle,
                                 train=True, download=False, transform=transform)
    testset = mydatasets.RandomRotMNIST(root='./data', unit_angle=unit_angle,
                                  train=False, download=False, transform=transform)
else:
    raise NotImplementedError

# Setting the seed for a consistent split
random_seed = 543
np.random.seed(random_seed)
num_train = len(trainset)
indices = list(range(num_train))
np.random.shuffle(indices)
valid_size = 0.1
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          sampler=train_sampler)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                        sampler=valid_sampler)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=False)


def get_prob(logits):
    logits = logits - torch.max(logits, 1)[0].unsqueeze(1)
    logits = torch.exp(logits)
    prob = logits / logits.sum(dim=1).unsqueeze(1)
    return prob[:, 1]


def train_adv_epoch(epoch, adv, opt, dataloader, writer, tag='train', navib_model=None):
    # Function to train a single adversary
    train = tag == 'train'
    loss_logger = AverageMeter()
    correct = 0
    total = 0
    start_time = time.time()
    total_steps = len(dataloader.dataset)//args.adv_batch_size
    if args.adv_use_weighted_loss:
        weights = dataloader.dataset.get_confound_weights().to(device)
    if train:
        adv.train()
    else:
        adv.eval()
    for idx, (x, _, c) in enumerate(dataloader):
        x = x.to(device)
        c = c.to(device)
        if navib_model is not None:
            if args.model_name == 'MNIST_EncDec':
                _, latent, _ = navib_model(x)
            else:
                _, latent = navib_model(x)
            x = latent
        logits = adv(x)
        
        pred = torch.argmax(logits, 1)
        correct += torch.sum(pred == c).item()
        total += x.size(0)
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
    accuracy = correct * 100.0 / total
    print(adv.name, tag, 'acc :', accuracy)
    writer.add_scalar(adv.name + '_loss/' + tag, loss_logger.avg, epoch)
    writer.add_scalar(adv.name + '_acc/' + tag, accuracy, epoch)


def train_adv(adv, opt, trainloader, valloader, testloader, writer, navib_model=None):
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.65)
    for epoch in range(args.adv_num_epochs):
        train_adv_epoch(epoch, adv, opt, trainloader, writer, tag='train', navib_model=navib_model)
        train_adv_epoch(epoch, adv, opt, valloader, writer, tag='val', navib_model=navib_model)
        train_adv_epoch(epoch, adv, opt, testloader, writer, tag='test', navib_model=navib_model)
        lr_scheduler.step()
    # Save the model 
    print('Saving model')
    path_ = os.path.join(model_path, adv.name + '.pth')
    torch.save(adv.state_dict(), path_)
    print('Training done!')


def train_adversaries(trainloader, valloader, testloader, writer, navib_model=None):
    if navib_model is None:
        dummy_x, _, _ = trainset.__getitem__(0)
        input_dim = dummy_x.size(0)
    else:
        navib_model.eval()
        input_dim = args.latent_dim
    output_dim = 5
    hidden_dim = args.adv_hidden_dim

    hidden_layers=1
    if navib_model is None:
        name = 'Baseline_Adv'
    else:
        name = navib_model.name + '_New_Adv'
    adv = models.Adv(name + str(hidden_layers), input_dim=input_dim, output_dim=output_dim,
                     hidden_dim=hidden_dim, hidden_layers=hidden_layers).to(device)
    opt = optim.Adam(adv.parameters(), lr=args.adv_lr, weight_decay=1e-4)
    train_adv(adv, opt, trainloader, valloader, testloader, writer, navib_model)


# Next the model is trained using different regularizers.
def icl(mu, labels):
    diff = mu.unsqueeze(0) - mu.unsqueeze(1)
    diff = diff.norm(dim = -1)
    sq_loss = diff.pow(2)
    lap_kernel = torch.exp(args.icl_a - args.icl_b * diff)
    
    s = torch.zeros_like(labels).float()
    c = 5
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


def mmd_loss(mu, labels):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.pow(2).sum(-1)

    s = torch.zeros_like(labels).float()
    c = 5
    for i in range(5):
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


def mmd_lap_loss(mu, labels):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.norm(dim = -1)
    lap_kernel = torch.exp(-diff/args.mmd_lap_p)

    s = torch.zeros_like(labels).float()
    c = 5
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


def train_disc(epoch, adv, opt, trainloader, writer, navib_model):
    assert(navib_model is not None)
    navib_model.eval()
    train_adv_epoch(epoch, adv, opt, trainloader, writer, tag='train',
                    navib_model=navib_model)
    print('Disc epoch!')


def navib_epoch(epoch, model, opt, dataloader, writer, tag='train',
                disc=None, disc_opt=None):
    train = tag == 'train'
    loss_logger = AverageMeter() # Total loss logger
    pred_loss_logger = AverageMeter()
    comp_loss_logger = AverageMeter()
    latent_logger = AverageMeter()
    recons_logger = AverageMeter()
    if train:
        if args.comp_type == 'adv_training':
            assert disc is not None
            assert disc_opt is not None
            train_disc(epoch, disc, disc_opt, dataloader, writer, model)
            disc.eval()
        model.train()
    else:
        model.eval()
    total_steps = len(dataloader.dataset)//args.batch_size
    
    y_correct = 0
    y_total = 0
    
    start_time = time.time()

    for idx, (x, y, c) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        c = c.to(device)

        if args.model_name == 'MNIST_EncDec':
            pred_logits, latent, latent2 = model(x)
            recons = model.decoder(latent, latent2)
            recons_loss = F.mse_loss(recons, x)
        else:
            recons_loss = torch.tensor(0).to(device)
            pred_logits, latent = model(x)

        pred_loss = F.cross_entropy(pred_logits, y)
        
        if args.comp_type == 'none':
            comp_loss = torch.tensor(0).to(device)
        elif args.comp_type == 'icl':
            comp_loss = icl(latent, c)
        elif args.comp_type == 'mmd_loss':
            comp_loss = mmd_loss(latent, c)
        elif args.comp_type == 'mmd_lap':
            comp_loss = mmd_lap_loss(latent, c)
        elif args.comp_type == 'adv_training':
            if train:
                logits = disc(latent)
                comp_loss = -1 * F.cross_entropy(logits, c)
            else:
                comp_loss = torch.tensor(0).to(device)
        else:
            raise NotImplementedError
        
        loss = args.pred_lambda * pred_loss + args.alpha * comp_loss
        loss += args.beta * recons_loss
        
        pred = torch.argmax(pred_logits, 1)
        y_correct += torch.sum(pred == y).item()
        y_total += x.size(0)
        
        # Log the losses
        loss_logger.update(loss.item())
        pred_loss_logger.update(pred_loss.item())
        comp_loss_logger.update(comp_loss.item())
        latent_logger.update(latent.norm(dim=-1).mean())
        recons_logger.update(recons_loss.item())
        
        if idx % args.log_step == 0:
            log_loss(epoch, args.num_epochs, idx, total_steps, loss, start_time)
            start_time = time.time()
        
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    model_name = 'navib_'
    accuracy = y_correct * 100.0 / y_total
    print(tag, 'accuracy:', accuracy)
    writer.add_scalar(model_name + 'acc/' + tag, accuracy, epoch)
    writer.add_scalar(model_name + 'pred_loss/' + tag, pred_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'comp_loss/' + tag, comp_loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'mu/' + tag, latent_logger.avg, epoch)
    writer.add_scalar(model_name + 'loss/' + tag, loss_logger.avg, epoch)
    writer.add_scalar(model_name + 'recons/' + tag, recons_logger.avg, epoch)
    return accuracy


def train_navib(model, opt, trainloader, valloader, testloader, writer):
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5)
    best_val_acc = 0
    if args.comp_type == 'adv_training':
        disc = models.Adv('Disc', input_dim=args.latent_dim, output_dim=5,
                          hidden_dim=args.adv_hidden_dim, hidden_layers=args.disc_hidden_layers).to(device)
        disc_opt = optim.Adam(disc.parameters(), lr=args.disc_lr)
    for epoch in range(1, args.num_epochs + 1):
        if args.comp_type == 'adv_training':
            navib_epoch(epoch, model, opt, trainloader, writer, tag='train',
                        disc=disc, disc_opt=disc_opt)
        else:
            navib_epoch(epoch, model, opt, trainloader, writer, tag='train')

        val_acc = navib_epoch(epoch, model, opt, valloader, writer, tag='val')
        test_Acc = navib_epoch(epoch, model, opt, testloader, writer, tag='test')

        if val_acc > best_val_acc:
            name = 'Navib_best_val_acc'
            model.name = name
            path = os.path.join(model_path, name + '.pth')
            torch.save(model.state_dict(), path)
            best_val_acc = val_acc
            message = 'Best val acc {}. test_acc{}. Saving model {}'.format(
                best_val_acc,test_Acc, path)
            print(message)
            logf.write(message + '\n')
        if epoch % args.save_step == 0:
            name = 'Navib_ckpt_{}'.format(epoch)
            model.name = name
            path = os.path.join(model_path, name + '.pth')
            torch.save(model.state_dict(), path)
            print('Saving checkpoint {}'.format(path))

        lr_scheduler.step()
        if args.alpha < args.alpha_max:
            args.alpha *= args.alpha_gamma
    name = 'Navib'
    model.name = name
    path = os.path.join(model_path, name + '.pth')
    torch.save(model.state_dict(), path)


def navib(trainloader, valloader, testloader, writer):
    dummy_x, _, _ = trainset.__getitem__(0)
    if args.model_name == 'MNIST_EncDec':
        model = models.MNIST_EncDec(name='Navib', latent_dim=args.latent_dim,
                                    latent_dim2=args.latent_dim2).to(device)
    else:
        raise NotImplementedError
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_navib(model, opt, trainloader, valloader, testloader, writer)


if not args.only_adv_training:
    navib(trainloader, valloader, testloader, writer)


def run_adversaries(trainloader, valloader, testloader, writer, model_name='Navib'):
    dummy_x, _, _ = trainset.__getitem__(0)
    name = model_name
    path = os.path.join(model_path, name + '.pth')
    if args.model_name == 'MNIST_EncDec':
        model = models.MNIST_EncDec(name=model_name, latent_dim=args.latent_dim,
                                    latent_dim2=args.latent_dim2).to(device)
    else:
        raise NotImplementedError
    model.load_state_dict(torch.load(path))
    model.name = name
    train_adversaries(trainloader, valloader, testloader, writer, model)



print('Running adversaries')
# run_adversaries(trainloader, valloader, testloader, writer, model_name='Navib')
run_adversaries(trainloader, valloader, testloader, writer,
                model_name='Navib_best_val_acc')
print('Done adversaries')