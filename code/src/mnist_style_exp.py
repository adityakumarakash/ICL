import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os, sys, time, shutil
import torch
import torch.nn.functional as F
import torch.optim as optim

from src import model as models
from src.average_meter import AverageMeter

from MulticoreTSNE import MulticoreTSNE as TSNE
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='mnist_style')
parser.add_argument('--dataset_name', type=str, default='MNIST')
parser.add_argument('--model_name', type=str, default='IVAE')
parser.add_argument('--result_path', type=str, default='result', help='output path')
parser.add_argument('--data_path', type=str, default='data', help='path for data')

# Training
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--embed_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed to use')
parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=20, help='step size for saving trained models')

parser.add_argument('--flag_retrain', default=False, action='store_true', help='Re train')
parser.add_argument('--flag_reg', default=False, action='store_true', help='Regularizer')
parser.add_argument('--flag_plot', default=False, action='store_true', help='Plot')

parser.add_argument('--latent_dim', type=int, default=8)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.01)

parser.add_argument('--comp_lambda', type=float, default=1.0, 
                    help='Additional compression lambda argument')
parser.add_argument('--comp_gamma', type=float, default=1.25, 
                    help='The multiplicative factor of comp_lambda')
parser.add_argument('--comp_lambda_max', type=float, default=1.0, 
                    help='Maximum value compression lambda takes.')

parser.add_argument('--comp_type', type=str, default='icl_max', 
                    help='Choose from none, kl, ot_pairwise, mmd_loss, icl, mmd_lap, adv')

parser.add_argument('--icl_a', type=float, default=1.0, help='Argument a for ICL')
parser.add_argument('--icl_b', type=float, default=1.0, help='Argument b for ICL')
parser.add_argument('--mmd_lap_p', type=float, default=1.0, 
                    help='Argument for mmd laplacian')

parser.add_argument('--adv_num_epochs', type=int, default=100, help='Num of epochs for adv')
parser.add_argument('--adv_lr', type=float, default=1e-3, help='Adv learning rate')
parser.add_argument('--use_bce', default=False, action='store_true', help='use bce for reconstruction')
parser.add_argument('--run_multi_adv', default=False, action='store_true', 
                    help='Runs multiple adversaries at the end')
parser.add_argument('--disc_lr', type=float, default=1e-3,
                    help='LR for the adv training based regularizer')
parser.add_argument('--debug_tag', type=str, default='debug')

# Model parameters
args = parser.parse_args()

device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
          'beta', args.beta, 
          'comp_lambda', args.comp_lambda, 
          'comp_gamma', args.comp_gamma,
          'comp_lambda_max', args.comp_lambda_max,
          'seed', args.seed]

if args.comp_type == 'icl':
    add_params = ['icl_a', args.icl_a, 
                  'icl_b', args.icl_b]
    params.extend(add_params)
    
if args.comp_type == 'mmd_lap':
    add_params = ['lap_p', args.mmd_lap_p]
    params.extend(add_params)
    
if args.comp_type == 'icl_max':
    add_params = ['icl_p', args.mmd_lap_p]
    params.extend(add_params)
    
if args.use_bce:
    params.extend(['bce_loss'])

params_str = '_'.join([str(x) for x in params])

if args.debug_tag is not None:
    params_str = args.debug_tag + '_' + params_str

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


make_dir(args.result_path)
make_dir(output_path)
make_dir(sample_path)
make_dir(run_path)
logf=open(log_path, 'w')
make_dir(writer_path, rm=True)
writer=SummaryWriter(comment=model_name, log_dir=writer_path)
make_dir(model_path)


# Logging helper functions 
def log_loss(epoch, step, total_step, loss, start_time):
    loss=loss.cpu().data.numpy()
    message='Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, time: {:.4f}s'.format(
        epoch, args.num_epochs, step, total_step, loss, time.time() - start_time)
    logf.write(message + '\n')
    print(message)
    

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
valset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)


# Setting the seed for a consistent split
random_seed=543
np.random.seed(random_seed)
num_train = len(trainset)
indices = list(range(num_train))
np.random.shuffle(indices)
print(indices[:20])
valid_size = 0.1
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                          sampler=train_sampler)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                        sampler=valid_sampler)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)


latent_dim = args.latent_dim
nuisance_feature_dim = 10
input_channels = 1
vae = models.SimpleVAE(input_channels, latent_dim, nuisance_feature_dim).to(device)
optimizer = optim.Adam(vae.parameters(), lr=args.lr)


# Loss functions
def get_nuisance_feature(classes):
    one_hot = torch.zeros(classes.size(0), 10).to(device)
    one_hot.scatter_(1, classes.unsqueeze(1), 1)
    return one_hot


def reconstruction_loss(x, xhat):
    if args.use_bce:
        return F.binary_cross_entropy_with_logits(xhat, x, reduction='sum')/x.size(0)
    else:
        return F.mse_loss(x, xhat, reduction='sum')/x.size(0)


def kl_loss(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def compression_loss(mu, logvar):
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
    return loss.mean()


def wasserstein_pairwise_loss(mu, logvar):
    # Here we approximate the wasserstein loss using the pairwise wassersteins
    # This uses the closed form for the normals
    sigma = torch.exp(0.5 * logvar)
    diff_sigma = sigma.unsqueeze(0) - sigma.unsqueeze(1)
    diff_mu = mu.unsqueeze(0) - mu.unsqueeze(1)
    loss = diff_mu.pow(2).sum(dim=-1).mean()
    loss += diff_sigma.pow(2).sum(dim=-1).mean()
    return loss


def icl(mu, labels, logvar):
    diff = mu.unsqueeze(0) - mu.unsqueeze(1)
    diff = diff.norm(dim = -1)
    sq_loss = diff.pow(2)   # Squared loss for attraction
    lap_kernel = torch.exp(args.icl_a - args.icl_b * diff)  # Laplacian kernel for repulsion
    
    s = torch.zeros_like(labels).float()
    for i in range(10):
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            continue
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)
    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    loss = (sq_loss * s_prod)[c_diff == True].sum()
    loss += 9 * (lap_kernel * s_prod)[c_diff == False].sum()

    # The sigma terms are added since they are also present in the KL loss.
    sigma = torch.exp(0.5 * logvar)
    diff_sigma = sigma.unsqueeze(0) - sigma.unsqueeze(1)
    loss += diff_sigma.pow(2).sum(-1).mean()
    
    return loss


def mmd_loss(mu, labels, logvar):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.pow(2).sum(-1)

    s = torch.zeros_like(labels).float()
    for i in range(10):
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            continue
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)

    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    diff = diff * s_prod
    loss = diff[c_diff == True].sum()
    loss += -9*diff[c_diff == False].sum()
    
    sigma = torch.exp(0.5 * logvar)
    diff_sigma = sigma.unsqueeze(0) - sigma.unsqueeze(1)
    loss += diff_sigma.pow(2).sum(-1).mean()
    
    return loss


def mmd_lap_loss(mu, labels, logvar):
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)
    diff = diff.norm(dim = -1)
    lap_kernel = torch.exp(-diff/args.mmd_lap_p)

    s = torch.zeros_like(labels).float()
    for i in range(10):
        index = labels == i
        n = index.sum().item()
        if n < 1e-9:
            continue
        s[index] = 1.0/n
    s_prod = s.unsqueeze(1) * s.unsqueeze(0)

    c_diff = labels.unsqueeze(1) != labels.unsqueeze(0)
    lap_kernel = lap_kernel * s_prod
    loss = -lap_kernel[c_diff == True].sum()
    loss += 9*lap_kernel[c_diff == False].sum()
    
    sigma = torch.exp(0.5 * logvar)
    diff_sigma = sigma.unsqueeze(0) - sigma.unsqueeze(1)
    loss += diff_sigma.pow(2).sum(-1).mean()
    
    return loss


def disc_train_epoch(model, vae, epoch, optimizer, dataloader, writer):
    # model is the discriminator
    # vae is the encoder model
    # optimizer is for the discriminator
    vae.eval()
    start_time = time.time()
    total_step = len(dataloader.dataset) // args.batch_size
    model.train()
    loss_logger = AverageMeter()
    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        latent, _ = vae.encode(images)
        pred = model(latent)
        loss = F.cross_entropy(pred, labels)
        loss_logger.update(loss.item())
        if idx % args.log_step == 0:
            log_loss(epoch, idx, total_step, loss, start_time)
            start_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.add_scalar('DiscLoss/train', loss_logger.avg, epoch)
    model.eval()


def train_epoch(epoch, model, dataloader, optimizer, writer, disc=None, disc_opt=None):
    start_time = time.time()
    total_step = len(dataloader.dataset) // args.batch_size
    loss_logger = AverageMeter()
    prior_loss_logger = AverageMeter()
    recons_loss_logger = AverageMeter()
    comp_loss_logger = AverageMeter()
    sigma_logger = AverageMeter()
    mu_logger = AverageMeter()

    if args.comp_type == 'adv_training':
        # One epoch of training discriminator done.
        disc_train_epoch(disc, model, epoch, disc_opt, dataloader, writer)
        disc.eval()
    
    model.train()
    for idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        nuisance_features = get_nuisance_feature(labels)
        recons, mu, logvar = model(images, nuisance_features)
        latent = model.reparameterize(mu, logvar)
        
        # For understanding the latent space behaviour
        sigma_norm = torch.exp(0.5*logvar).norm(dim=-1).mean()
        sigma_logger.update(sigma_norm.item())
        mu_norm = mu.norm(dim=-1).mean()
        mu_logger.update(mu_norm.item())
        
        # Losses
        # reconstruction loss
        recons_loss = reconstruction_loss(images, recons)
        # KL loss for VAE
        prior_loss = kl_loss(mu, logvar)
        
        loss =  args.beta * prior_loss + (1 + args.alpha) * recons_loss
        
        # compression loss
        if args.comp_lambda > 1e-9:
            if args.comp_type == 'kl':
                comp_loss = compression_loss(mu, logvar)
            elif args.comp_type == 'ot_pairwise':
                comp_loss = wasserstein_pairwise_loss(mu, logvar)
            elif args.comp_type == 'mmd_loss':
                comp_loss = mmd_loss(mu, labels, logvar)
            elif args.comp_type == 'icl':
                comp_loss = icl(mu, labels, logvar)
            elif args.comp_type == 'mmd_lap':
                comp_loss = mmd_lap_loss(mu, labels, logvar)
            elif args.comp_type == 'none':
                comp_loss = torch.tensor(0.0).to(device)
            elif args.comp_type == 'adv_training':
                pred = disc(mu)
                comp_loss = -1 * F.cross_entropy(pred, labels)
            else:
                raise NotImplementedError

            loss += args.comp_lambda * args.alpha * comp_loss
            comp_loss_logger.update(comp_loss.item())
        
        loss_logger.update(loss.item())
        prior_loss_logger.update(prior_loss.item())
        recons_loss_logger.update(recons_loss.item())
        
        if idx % args.log_step == 0:
            log_loss(epoch, idx, total_step, loss, start_time)
            start_time = time.time()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Update the logs
    writer.add_scalar('Loss/train', loss_logger.avg, epoch)
    writer.add_scalar('Recons_Loss/train', recons_loss_logger.avg, epoch)
    writer.add_scalar('Comp_Loss/train', comp_loss_logger.avg, epoch)
    writer.add_scalar('Sigma/train', sigma_logger.avg, epoch)
    writer.add_scalar('Mu/train', mu_logger.avg, epoch)


def test_epoch(epoch, model, dataloader, optimizer, writer, tag='test'):
    model.eval()
    loss_logger = AverageMeter()
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        nuisance_features = get_nuisance_feature(labels)
        recons, mu, logvar = model(images, nuisance_features)
        loss = reconstruction_loss(images, recons)
        loss_logger.update(loss.item())
    writer.add_scalar('Recons_Loss/'+tag, loss_logger.avg, epoch)


def main(model, trainloader, valloader, testloader, opt, writer):
    #lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.65)
    if args.comp_type == 'adv_training':
        num_hidden = 3
        disc = models.Adv(name='Adv' + str(num_hidden), input_dim=latent_dim,
                          output_dim=10, hidden_layers=num_hidden).to(device)
        disc_opt = optim.Adam(disc.parameters(), lr=args.disc_lr)
    for epoch in range(args.num_epochs):
        if args.comp_type == 'adv_training':
            train_epoch(epoch, model, trainloader, opt, writer, disc, disc_opt)
        else:
            train_epoch(epoch, model, trainloader, opt, writer)
        test_epoch(epoch, model, valloader, opt, writer, tag='val')
        test_epoch(epoch, model, testloader, opt, writer, tag='test')
        if args.comp_lambda < args.comp_lambda_max:
            args.comp_lambda *= args.comp_gamma
        #lr_scheduler.step()
    print('Saving the model')
    torch.save(model.state_dict(), net_path)


main(vae, trainloader, valloader, testloader, optimizer, writer)


#### Train adversary ####
# Train a model to predict the digit from the latent rep
# Assuming our encoder is deterministic for now, we train a simple model to predict
# which digit is encoded
def train_adversary(model, vae, optimizer, num_epochs, writer, trainloader, valloader, testloader):
    vae.eval()
    def get_accuracy(model, vae, dataloader):
        model.eval()
        total = 0
        correct = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            latent, _ = vae.encode(images)
            pred_logits = model(latent)
            pred_labels = torch.argmax(pred_logits, 1)
            total += images.size(0)
            correct += torch.sum(pred_labels == labels).item()
        return correct*100.0/total

    def train_epoch(model, vae, epoch, optimizer, dataloader, writer):
        start_time = time.time()
        total_step = len(dataloader.dataset) // args.batch_size
        model.train()
        loss_logger = AverageMeter()
        for idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            latent, _ = vae.encode(images)
            pred = model(latent)
            loss = F.cross_entropy(pred, labels)
            loss_logger.update(loss.item())
            
            if idx % args.log_step == 0:
                log_loss(epoch, idx, total_step, loss, start_time)
                start_time = time.time()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if args.run_multi_adv:
            writer.add_scalar(model.name + '_Loss/train', loss_logger.avg, epoch)
            writer.add_scalar(model.name + '_Acc/train', get_accuracy(model, vae, dataloader), epoch)
        else:
            writer.add_scalar('AdvLoss/train', loss_logger.avg, epoch)
            writer.add_scalar('AdvAcc/train', get_accuracy(model, vae, dataloader), epoch)

    def test_epoch(model, vae, epoch, dataloader, writer, tag='test'):
        model.eval()
        if args.run_multi_adv:
            writer.add_scalar(model.name + '_Acc/'+tag, get_accuracy(model, vae, dataloader), epoch)
        else:
            writer.add_scalar('AdvAcc/'+tag, get_accuracy(model, vae, dataloader), epoch)
    
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    for epoch in range(num_epochs):
        train_epoch(model, vae, epoch, optimizer, trainloader, writer)
        test_epoch(model, vae, epoch, valloader, writer, tag='val')
        test_epoch(model, vae, epoch, testloader, writer, tag='test')
        lr_scheduler.step()
    print('Test accuracy adversarial ', get_accuracy(model, vae, testloader))


def start_adversarial_training(vae, writer, trainloader, valloader, testloader):
    num_hidden = 3
    adv = models.Adv(name='Adv' + str(num_hidden), input_dim=latent_dim,
                     output_dim=10, hidden_layers=num_hidden).to(device)
    adv_optimizer = optim.Adam(adv.parameters(), lr=args.adv_lr)
    train_adversary(adv, vae, adv_optimizer, args.adv_num_epochs, writer, trainloader,
                    valloader, testloader)


print('Running Adversarial training now')
start_adversarial_training(vae, writer, trainloader, valloader, testloader)
print('Adversarial training done')


####### tSNE Plots  ########
# Create tSNE plot
def get_data(vae, dataloader):
    vae.eval()
    X = []
    Y = []
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        latent, _ = vae.encode(images)
        X.append(latent)
        Y.append(labels)
    X = torch.cat(X).cpu().detach().numpy()
    Y = torch.cat(Y).cpu().detach().numpy()
    return X, Y


def get_samples(X, Y):
    idx = np.random.permutation(X.shape[0])
    nsamples = min(10000, X.shape[0])
    selected = idx[0:nsamples]
    return X[selected, :], Y[selected]


def vis_tsne(X, Y):
    embeddings = TSNE(n_jobs=4).fit_transform(X)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    plt.figure(figsize=(10, 10))
    plt.scatter(vis_x, vis_y, c=Y, cmap=plt.cm.get_cmap("jet", 10), marker='.')
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    fig = plt.gcf()

    # Save the figure
    figpath = os.path.join(writer_path, 'tSNE.png')
    fig.savefig(figpath, bbox_inches='tight')
    print('Saved', figpath)


def make_tsne_plots(vae, dataloader):
    X, Y = get_data(vae, dataloader)
    X, Y = get_samples(X, Y)
    vis_tsne(X, Y)


print('Generating tSNE plots')
make_tsne_plots(vae, testloader)