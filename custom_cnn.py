from torch import nn

params = {
    'FashionMNIST0.3': {
        'num_epochs': 10,
        'input_dimension_size': 28,
        'conv_layers': 1,
        'conv_dimensions': 8,
        'fc_layers': 2,
        'fc_neurons': 1000,
        'rgb': False
    },
    'FashionMNIST0.6': {
        'num_epochs': 10,
        'input_dimension_size': 28,
        'conv_layers': 1,
        'conv_dimensions': 6,
        'fc_layers': 5,
        'fc_neurons': 1000,
        'rgb': False
    },
    'CIFAR10': {
        'num_epochs': 10,
        'input_dimension_size': 32,
        'conv_layers': 1,
        'conv_dimensions': 8,
        'fc_layers': 2,
        'fc_neurons': 1000,
        'rgb': True
    }
}


class CustomCNN(nn.Sequential):
    def __init__(self, input_dimension_size, conv_layers, conv_dimensions, fc_layers, fc_neurons, rgb):
        # Setup related stuff
        layers = []
        output_dim = input_dimension_size - conv_layers*conv_dimensions + conv_layers
        rgb_channels = 3 if rgb else 1
        
        for i in range(conv_layers):
            conv = nn.Conv2d(rgb_channels, rgb_channels, conv_dimensions)
            layers.append(conv)
            layers.append(nn.LeakyReLU())

        layers.append(nn.Flatten())
        
        if fc_layers > 1:
            layers.append(nn.Linear(rgb_channels * output_dim**2, fc_neurons))
            layers.append(nn.LeakyReLU())
        else:
            layers.append(nn.Linear(rgb_channels * output_dim**2, 4))

        if fc_layers > 2:
            for i in range(fc_layers - 2):
                layers.append(nn.Linear(fc_neurons, fc_neurons))
                layers.append(nn.LeakyReLU())

        if fc_layers >= 2:
            layers.append(nn.Linear(fc_neurons, 4))

        super().__init__(*layers)