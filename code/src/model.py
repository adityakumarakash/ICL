import torch

from torch import nn
from torch.autograd import Variable

class Adv(nn.Module):
    # New adversarial model for baseline which uses ReLU activation
    # The output would be logits
    def __init__(self, name='Adv', input_dim=64, output_dim=10, hidden_dim=64, hidden_layers=3):
        super(Adv, self).__init__()
        self.name = name
        layers = []
        prev_dim = input_dim
        self.output_dim = output_dim
        for i in range(0, hidden_layers + 1):
            if i == 0:
                prev_dim = input_dim
            else:
                prev_dim = hidden_dim
            
            if i == hidden_layers:
                layers.append(nn.Linear(prev_dim, output_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(64))  # This is different from the previous adv
                layers.append(nn.ReLU())
        self.adv = nn.Sequential(*layers)
        
    def forward(self, x):
        output = self.adv(x)
        if self.output_dim == 1:
            output = output.squeeze()
        return output


class SimpleVAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64, feature_dim=0):
        super(SimpleVAE, self).__init__()
        self.name = 'SimpleVAE'
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        # Encoder layers
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, self.latent_dim)
        self.fc32 = nn.Linear(512, self.latent_dim)

        # Decoder layers
        self.fc4 = nn.Linear(self.latent_dim + self.feature_dim, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 28*28)

        self.relu = nn.ReLU()

    def encode(self, x):
        fc1 = self.relu(self.fc1(x.view(-1, 28*28)))
        fc2 = self.relu(self.fc2(fc1))
        return self.fc31(fc2), self.fc32(fc2)

    def decode(self, x):
        fc4 = self.relu(self.fc4(x))
        fc5 = self.relu(self.fc5(fc4))
        fc6 = self.fc6(fc5)
        out = torch.sigmoid(fc6.view(-1, 1, 28, 28))
        return out

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, x, c):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        inp = torch.cat([z, c], 1)
        return self.decode(inp), mu, logvar

    
class CVAE(nn.Module):
    def __init__(self, input_channels=1, latent_dim=64, feature_dim=0):
        super(CVAE, self).__init__()

        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        # Encoder layers
        self.fc1 = nn.Linear(28*28 + self.feature_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, self.latent_dim)
        self.fc32 = nn.Linear(512, self.latent_dim)

        # Decoder layers
        self.fc4 = nn.Linear(self.latent_dim + self.feature_dim, 512)
        self.fc5 = nn.Linear(512, 512)
        self.fc6 = nn.Linear(512, 28*28)

        self.relu = nn.ReLU()

    def encode(self, x):
        fc1 = self.relu(self.fc1(x))
        fc2 = self.relu(self.fc2(fc1))
        return self.fc31(fc2), self.fc32(fc2)

    def decode(self, x):
        fc4 = self.relu(self.fc4(x))
        fc5 = self.relu(self.fc5(fc4))
        fc6 = self.fc6(fc5)
        out = torch.sigmoid(fc6.view(-1, 1, 28, 28))
        return out

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, x, c):
        x_cat = torch.cat([x.view(-1, 28*28), c], 1)
        mu, logvar = self.encode(x_cat)
        z = self.reparameterize(mu, logvar)
        inp = torch.cat([z, c], 1)
        return self.decode(inp), mu, logvar


class UAIModel(nn.Module):
    """
    There is one encoder, one decoder and one discriminator.
    This is replication of the baseline from tf from the dcmoyer/inv-rep repo uses ReLU activations.
    The final prediction output is logits.
    """
    def __init__(self, name='Navib', input_dim=121, latent_dim=64, hidden_dim=64):
        super(UAIModel, self).__init__()

        self.name = name
        self.latent_dim = latent_dim

        # Encoder layers
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, self.latent_dim)  # e1
        self.fc22 = nn.Linear(hidden_dim, self.latent_dim)  # e2

        # Decoder layers
        self.fc3 = nn.Linear(self.latent_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # y-predictor
        self.fc5 = nn.Linear(self.latent_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 2)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)

    def encode(self, x):
        fc1 = self.relu(self.fc1(x))
        e1 = self.tanh(self.fc21(fc1))
        e2 = self.tanh(self.fc22(fc1))
        return e1, e2

    def decode(self, x):
        fc3 = self.relu(self.fc3(x))
        fc4 = self.fc4(fc3)
        return fc4

    def predict(self, x):
        fc5 = self.relu(self.fc5(x))
        fc6 = self.fc6(fc5)
        return fc6

    def forward(self, x):
        e1, e2 = self.encode(x)
        e1_noisy = self.dropout(e1)
        recons = self.decode(torch.cat([e1_noisy, e2], dim=1))
        pred = self.predict(e1)
        return recons, pred, e1, e2


class UAIDisentangler(nn.Module):
    def __init__(self, latent_dim=64):
        super(UAIDisentangler, self).__init__()
        self.latent_dim = latent_dim
        self.e1_fc1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.e2_fc1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.tanh = nn.Tanh()

    def forward(self, e1, e2):
        return self.tanh(self.e2_fc1(e2)), self.tanh(self.e1_fc1(e1))


class BaselineVAE(nn.Module):
    """
    There is one encoder, one decoder and one discriminator.
    This is replication of the baseline from tf from the dcmoyer/inv-rep repo uses ReLU activations.
    The final prediction output is logits.
    """
    def __init__(self, name='Navib', input_dim=121, latent_dim=64, feature_dim=0):
        super(BaselineVAE, self).__init__()
        
        self.name = name
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        # Encoder layers
        hidden_dim = 64
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, self.latent_dim) 
        self.fc22 = nn.Linear(hidden_dim, self.latent_dim) 

        # Decoder layers
        self.fc3 = nn.Linear(self.latent_dim + self.feature_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        # y-predictor
        self.fc5 = nn.Linear(self.latent_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 2)  # One logit prediction for regression like task
        
        self.relu = nn.ReLU()

    def encode(self, x):
        fc1 = self.relu(self.fc1(x))
        mu = self.fc21(fc1)
        logvar = self.fc22(fc1)
        return mu, logvar

    def decode(self, x):
        fc3 = self.relu(self.fc3(x))
        fc4 = self.fc4(fc3)
        return fc4
    
    def predict(self, x):
        fc5 = self.relu(self.fc5(x))
        fc6 = self.fc6(fc5)
        return fc6

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, x, c):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        inp = torch.cat([z, c.float()], 1)
        recons = self.decode(inp)
        pred = self.predict(z)
        return recons, pred, mu, logvar


class OriginalBaselineAdv(nn.Module):
    def __init__(self, name='Adv', input_dim=10, output_dim=1, hidden_dim=64, hidden_layers=0):
        super(OriginalBaselineAdv, self).__init__()
        self.name = name
        layers = []
        prev_dim = input_dim
        for i in range(0, hidden_layers + 1):
            if i == 0:
                prev_dim = input_dim
            else:
                prev_dim = hidden_dim
            
            if i == hidden_layers:
                layers.append(nn.Linear(prev_dim, output_dim))
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.Tanh())
        self.adv = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.adv(x).squeeze()


class BaselineAdv(nn.Module):
    # New adversarial model for baseline which uses ReLU activation
    # The output would be logits
    def __init__(self, name='Adv', input_dim=10, output_dim=2, hidden_dim=64, hidden_layers=0):
        super(BaselineAdv, self).__init__()
        self.name = name
        layers = []
        prev_dim = input_dim
        self.output_dim = output_dim
        for i in range(0, hidden_layers + 1):
            if i == 0:
                prev_dim = input_dim
            else:
                prev_dim = hidden_dim
            
            if i == hidden_layers:
                layers.append(nn.Linear(prev_dim, output_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
        self.adv = nn.Sequential(*layers)
        
    def forward(self, x):
        output = self.adv(x)
        if self.output_dim == 1:
            output = output.squeeze()
        return output


class MNIST_FC(nn.Module):
    def __init__(self, name='FC_Model', input_dim=784, output_dim=10, latent_dim=32):
        super(MNIST_FC, self).__init__()
        self.name = name
        # Expect input of dim = 784
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.fc3 = nn.Linear(latent_dim, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, self.output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        inp = x.view(-1, self.input_dim)
        out1 = self.relu(self.bn1(self.fc1(inp)))
        out1mid = self.fc2(out1)
        out2 = self.relu(self.bn2(out1mid))
        out3 = self.relu(self.bn3(self.fc3(out2)))
        return self.fc4(out3), out2


class MNIST_Conv(nn.Module):
    def __init__(self, name='FC_Model', input_channels=3, output_dim=10, latent_dim=32):
        super(MNIST_Conv, self).__init__()
        self.name = name
        self.output_dim = output_dim
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(self.input_channels, 8, kernel_size=3, stride=1, padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(8*7*7, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.relu(self.max_pool2d(self.conv1(x)))
        out2 = self.relu(self.max_pool2d(self.conv2(out1)))
        latent = self.relu(self.fc1(out2.reshape(out2.size(0), -1)))
        out3 = self.relu(self.fc2(latent))
        return self.fc3(out3), latent


class MNIST_Rot(nn.Module):
    def __init__(self, name='FC_Model', input_channels=3, output_dim=10, latent_dim=30):
        super(MNIST_Rot, self).__init__()
        self.name = name
        self.output_dim = output_dim
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=5, stride=2, padding=2)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*14*14, latent_dim)
        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.fc2 = nn.Linear(latent_dim, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # pdb.set_trace()
        out1 = self.relu(self.bn_conv1(self.conv1(x)))

        latent = self.fc1(out1.reshape(out1.size(0), -1))

        out2 = self.relu(self.bn2(self.fc2(self.bn1(latent))))
        return self.fc3(out2), latent


class MNIST_EncDec(nn.Module):
    def __init__(self, name='FC_Model', input_channels=1, output_dim=10,
                 latent_dim=10, latent_dim2=20):
        super(MNIST_EncDec, self).__init__()
        self.name = name
        self.output_dim = output_dim
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=5, stride=2, padding=2)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 14 * 14, latent_dim)
        self.fc12 = nn.Linear(64 * 14 * 14, latent_dim2)

        self.bn1 = nn.BatchNorm1d(latent_dim)
        self.fc2 = nn.Linear(latent_dim, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

        self.dec_fc1 = nn.Linear(latent_dim + latent_dim2, 256*14*14)
        self.dec_bn1 = nn.BatchNorm1d(256*14*14)
        self.dec_up_sample = nn.Upsample(scale_factor=2)
        self.dec_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv1_bn = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.Conv2d(128, self.input_channels, kernel_size=1)
        self.dec_sigmoid = nn.Sigmoid()

    def forward(self, x):
        out1 = self.relu(self.bn_conv1(self.conv1(x)))
        out1 = out1.reshape(out1.size(0), -1)
        latent = self.fc1(out1)
        latent2 = self.fc12(out1)

        out2 = self.relu(self.bn2(self.fc2(self.bn1(latent))))
        return self.fc3(out2), latent, latent2

    def decoder(self, latent1, latent2):
        x = torch.cat([latent1, latent2], dim=1)
        x = self.relu(self.dec_bn1(self.dec_fc1(x)))
        x = x.reshape(x.size(0), 256, 14, 14)
        x = self.dec_up_sample(x)
        x = self.relu(self.dec_conv1_bn(self.dec_conv1(x)))
        x = self.dec_sigmoid(self.dec_conv2(x))
        return x


class FC_DetEnc(nn.Module):
    def __init__(self, input_dim=10, output_dim=10, hidden_dim=30, latent_dim=30):
        super(FC_DetEnc, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.name = 'Vib'
    
    def encode(self, x):
        in2 = self.relu(self.fc1(x))
        latent = self.relu(self.fc2(in2))
        return latent
    
    def predict(self, z):
        in4 = self.relu(self.fc3(z))
        out = self.fc4(in4)
        return out
        
    def forward(self, x):
        latent = self.encode(x)
        out = self.predict(latent)
        return out, latent