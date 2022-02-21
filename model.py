import torch

class FullyConnectedModel(torch.nn.Module):
	def __init__(self, in_features=784, out_features=10, num_layers=3):
		super(FullyConnectedModel, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.num_layers = num_layers
		self.neural_network = torch.nn.Sequential(*self.generate_network())

	def generate_network(self):
		layers = []
		layers_n = [self.in_features]+[self.out_features+i*(self.in_features - self.out_features)//self.num_layers for i in range(self.num_layers)][::-1]
		for i in range(1, self.num_layers):
			layers.append(torch.nn.Linear(layers_n[i-1], layers_n[i]))
			layers.append(torch.nn.ReLU())
		layers.append(torch.nn.Linear(layers_n[i], self.out_features))
		return layers

	def forward(self, x):
		return self.neural_network(x)

if __name__ == '__main__':
	model = FullyConnectedModel()
	print(model.neural_network)