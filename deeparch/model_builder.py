import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, use_bn, activation, pool_type):
        super().__init__()
        padding = kernel // 2
        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True) if activation=='relu' else nn.LeakyReLU(negative_slope=0.1, inplace=True))
        if pool_type == 'max':
            layers.append(nn.MaxPool2d(2))
        else:
            layers.append(nn.AvgPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class NASModel(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super().__init__()
        layers = []
        in_ch = 3
        for i in range(cfg['n_conv_blocks']):
            layers.append(ConvBlock(in_ch,
                                    cfg[f'filters_{i}'],
                                    cfg[f'kernel_{i}'],
                                    cfg[f'use_bn_{i}'],
                                    cfg['activation'],
                                    cfg[f'pool_{i}']))
            in_ch = cfg[f'filters_{i}']
        self.features = nn.Sequential(*layers)

        n_pools = sum(1 for i in range(cfg['n_conv_blocks']) if cfg[f'pool_{i}'] in ['max','avg'])
        spatial = max(1, 32 // (2 ** n_pools))
        flattened = in_ch * spatial * spatial

        fc_layers = []
        in_features = flattened
        for j in range(cfg['n_fc']):
            units = cfg[f'fc_units_{j}']
            fc_layers.append(nn.Linear(in_features, units))
            if cfg['fc_use_bn']:
                fc_layers.append(nn.BatchNorm1d(units))
            fc_layers.append(nn.ReLU(inplace=True) if cfg['activation']=='relu' else nn.LeakyReLU(0.1, inplace=True))
            if cfg['dropout'] > 0:
                fc_layers.append(nn.Dropout(cfg['dropout']))
            in_features = units
        fc_layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def build_model(cfg):
    return NASModel(cfg)