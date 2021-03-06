import torch
import torch.nn as nn
import torch.nn.functional as F

from src.adni_classification.utils import Swish


def get_activation_layer(name):
    return {
        'relu': nn.ReLU,
        'elu': nn.ELU,
        'softplus': nn.Softplus,
        'leaky_relu': nn.LeakyReLU,
        'swish': Swish
    }[name]


class UAIResNet(nn.Module):
    def __init__(self, in_depth, n_blocks, interm_depths, bottleneck=True, n_out_linear=None, dropout=0.):
        super(UAIResNet, self).__init__()
        self.name = 'Resnet'
        assert (len(n_blocks) == len(interm_depths))
        self.init_conv = nn.Conv3d(in_depth, interm_depths[0], kernel_size=7, stride=2, padding=3, bias=True)
        self.init_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=0)
        self.stages = nn.ModuleList([self._build_stage(n_blocks[i], interm_depths[max(0, i - 1)],
                                                       out_depth=interm_depths[i], stride=1 if i == 0 else 2,
                                                       bottleneck=bottleneck) for i in range(len(n_blocks))])
        self.pool_linear = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()

        # Second head for generating mask
        self.fc11 = nn.Linear(64, 64)
        self.fc12 = nn.Linear(64, 64)
        self.tanh = nn.Tanh()
        self.decoder = UAIDecoder(64*2)
        self.dropout = nn.Dropout(0.5)

        if n_out_linear is not None:
            self.output_head = MLP(interm_depths[-1], [interm_depths[-1] * 2], n_out_linear, dropout=dropout)
        else:
            self.output_head = None

        for m in self.modules():
            if type(m) in (nn.Conv3d, nn.Linear):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
            if type(m) is nn.BatchNorm3d:
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def encode(self, x):
        x = self.init_conv(x)
        x = self.init_pool(x)
        for s in self.stages:
            x = s(x)
        x = self.pool_linear(x)
        x = self.flatten(x)
        return self.tanh(self.fc11(x)), self.tanh(self.fc12(x))

    def decode(self, e1, e2):
        e2 = self.dropout(e2)
        return self.decoder(torch.cat([e1, e2], dim=1))

    def forward(self, x):
        e1, e2 = self.encode(x)
        if self.output_head is not None:
            logits, _ = self.output_head(e1)
        recons = self.decode(e1, e2)
        return logits, recons, e1, e2

    def _build_stage(self, n_blocks, in_depth, hidden_depth=None, out_depth=None, stride=2, dilation=1, batchnorm=True,
                     activation='swish', zero_output=True, bottleneck=True):
        if out_depth is None:
            out_depth = in_depth * stride
        blocks = [ResNetBlock(in_depth if i == 0 else out_depth, hidden_depth, out_depth,
                              stride=stride if i == 0 else 1,
                              dilation=dilation,
                              batchnorm=batchnorm,
                              activation=activation,
                              zero_output=zero_output,
                              bottleneck=bottleneck) for i in range(n_blocks)]
        return nn.Sequential(*blocks)

    def _build_linear_head(self, in_depth, hidden_depths, out_depth, activation='swish', batchnorm=True, dropout=0.):
        layers = []
        depths = [in_depth, *hidden_depths, out_depth]

        layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        layers.append(nn.Flatten())
        for i in range(len(depths) - 1):
            if dropout > 0.:
                layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(depths[i], depths[i + 1], bias=not batchnorm))
            if i != len(depths) - 2:
                if batchnorm:
                    layers.append(nn.BatchNorm1d(depths[i + 1], eps=1e-8))
                layers.append(get_activation_layer(activation)())

        return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, in_depth, hidden_depths, out_depth, activation='swish', batchnorm=True, dropout=0.):
        super(MLP, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.depths = [in_depth, *hidden_depths, out_depth]

        self.linear_layers = nn.ModuleList([])
        self.norm = nn.ModuleList([])
        self.act = nn.ModuleList([])

        for i in range(len(self.depths) - 1):
            self.linear_layers.append(nn.Linear(self.depths[i], self.depths[i + 1], bias=not batchnorm))
            if i != len(self.depths) - 2:
                if batchnorm:
                    self.norm.append(nn.BatchNorm1d(self.depths[i + 1], eps=1e-8))
                self.act.append(get_activation_layer(activation)())

    def forward(self, x):
        for i in range(len(self.depths) - 1):
            if self.dropout > 0.:
                x = F.dropout(x, self.dropout, self.training)
            x = self.linear_layers[i](x)
            if i != len(self.depths) - 2:
                if self.batchnorm:
                    x = self.norm[i](x)
                x = self.act[i](x)
            if i == len(self.depths) - 3:
                latent = x
        return x, latent
    
    def l1_norm(self):
        return sum([torch.norm(l.weight, 1) for l in self.linear_layers])
    
    def l2_norm(self):
        return sum([torch.norm(l.weight, 2) for l in self.linear_layers])


class ResNetBlock(nn.Module):
    def __init__(self, in_depth, hidden_depth=None, out_depth=None, stride=1, dilation=1, batchnorm=True,
                 activation='swish', zero_output=True, bottleneck=True):
        super(ResNetBlock, self).__init__()
        if out_depth is None:
            out_depth = in_depth * stride
        if stride > 1:
            self.shortcut_layer = nn.Conv3d(in_depth, out_depth, kernel_size=3, stride=stride,
                                            padding=1, dilation=dilation, bias=True)
        else:
            self.shortcut_layer = None

        layers = []
        if bottleneck:
            if hidden_depth is None:
                hidden_depth = in_depth // 4
            k_sizes = [3, 1, 3]
            depths = [in_depth, hidden_depth, hidden_depth, out_depth]
            paddings = [1, 0, 1]
            strides = [1, 1, stride]
            dilations = [dilation, 1, dilation]
        else:
            if hidden_depth is None:
                hidden_depth = in_depth
            k_sizes = [3, 3]
            depths = [in_depth, hidden_depth, out_depth]
            paddings = [1, 1]
            strides = [1, stride]
            dilations = [dilation, dilation]
        
        for i in range(len(k_sizes)):
            if batchnorm:
                layers.append(nn.BatchNorm3d(depths[i], eps=1e-8))
            layers.append(get_activation_layer(activation)())
            layers.append(nn.Conv3d(depths[i], depths[i + 1], k_sizes[i], padding=paddings[i],
                                    stride=strides[i], dilation=dilations[i], bias=False))
        
        self.layers = nn.Sequential(*layers)
        # for i, l in enumerate(self.layers):
        #     self.add_module("layer_{:d}".format(i), l)
        
    def forward(self, x):
        Fx = self.layers(x)
        if self.shortcut_layer is not None:
            x = self.shortcut_layer(x)
        return x + Fx


class UAIDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(UAIDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 32 * 3 * 4 * 3)
        self.bn = nn.BatchNorm1d(32 * 3 * 4 * 3)
        self.relu = nn.ReLU()
        activation='swish'
        self.convTrans1 = nn.Sequential(nn.ConvTranspose3d(32, 16, kernel_size=3,
                                                           stride=2),
                                        nn.BatchNorm3d(16),
                                        get_activation_layer(activation)())
        self.convTrans2 = nn.Sequential(nn.ConvTranspose3d(16, 8, kernel_size=3,
                                                           stride=2),
                                        nn.BatchNorm3d(8),
                                        get_activation_layer(activation)())
        self.convTrans3 = nn.Sequential(nn.ConvTranspose3d(8, 1, kernel_size=3,
                                                           stride=2))
        self.input_dim = (91, 109, 91)

    def forward(self, input):
        out1 = self.relu(self.bn(self.fc(input)))
        out2 = out1.reshape(out1.size(0), 32, 3, 4, 3)
        conv1 = self.convTrans1(out2)
        conv2 = self.convTrans2(conv1)
        conv3 = self.convTrans3(conv2)
        return F.interpolate(conv3, size=self.input_dim, mode='trilinear',
                             align_corners=True)


class UAIDisentangler(nn.Module):
    def __init__(self, latent_dim=64):
        super(UAIDisentangler, self).__init__()
        self.latent_dim = latent_dim
        self.e1_fc1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.e2_fc1 = nn.Linear(self.latent_dim, self.latent_dim)
        self.tanh = nn.Tanh()

    def forward(self, e1, e2):
        return self.tanh(self.e2_fc1(e2)), self.tanh(self.e1_fc1(e1))