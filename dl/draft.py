import numpy as np

class Activation(object):
    def __init__(self, name='linear'):
        self.name = name
        
    def __call__(self, inputs):
        if(self.name == 'linear'):
            return inputs
        elif(self.name == 'sigmoid'):
            exp = np.exp(-inputs)
            return 1 / (1 + exp)
        elif(self.name == 'tanh'):
            return np.tanh(inputs)

def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2

class Network(object):
    def __init__(self, name=None, input_shape=None):
        self.name = name
        self.input_shape = input_shape
        self.layers = []
        self.weights = []
        self.loss = mse

    def forward(self, inputs, params):
        outputs = inputs
        for layer in params:
            outputs = outputs @ layer['weights']
            outputs = layer['activation'](outputs)

        return outputs

    def add_layer(self, layer):
        if(len(self.layers) == 0):
            print(self.input_shape[-1])
            weights = np.random.rand(self.input_shape[-1], layer['output_channel'])
        else:
            weights = np.random.rand(self.layers[-1]['weights'].shape[1], layer['output_channel'])
           
        if('name' not in layer.keys()):
            layer_name = 'layer_%d' % len(self.layers)
        else:
            layer_name = layer['name']

        self.layers.append({
            'output_channel' : layer['output_channel'], 
            'activation': Activation(name=layer['activation']),
            'weights' : weights,
            'name' : layer_name
        })

    def compile(self, loss=mse, optimizer='sgd'):
        self.loss = mse

    def backward(self, inputs):
        eps = 1e-8
        for i, layer in enumerate(self.layers):

inputs = np.random.rand(30, 10)
net = Network(name='new', input_shape=[10])
net.add_layer({'output_channel' : 5, 'activation':'sigmoid'})
net.add_layer({'output_channel' : 2, 'activation':'sigmoid'})
print(net.forward(inputs))
