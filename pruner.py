import torch
from trainer import train

class ModelAugmenter():
	def __init__(self, model, seed, p):
		self.neural_network = model
		self.seed = seed
		self.p = p
		self.mask_indexes = {}
		self.initialize_weights()

	def init_weights(self, m):
		torch.manual_seed(self.seed)
		if type(m) == torch.nn.Linear:
			torch.nn.init.xavier_uniform_(m.weight)
		elif type(m) == torch.nn.Embedding:
			torch.nn.init.normal_(m.weight)

	def initialize_weights(self):
		self.neural_network.apply(self.init_weights)

	def mask(self, m, idx, tensor):
		prunable_features = self.mask_indexes[idx]
		mask = torch.ones_like(tensor)
		mask[prunable_features] = 0
		tensor = tensor*mask
		tensor = tensor.reshape(m.weight.data.shape)
		m.weight.data = tensor

	def apply_mask(self):
		for layer_id, layer in enumerate(self.neural_network.modules()):
			if type(layer) == torch.nn.Linear:
				tensor = layer.weight.data
				tensor = tensor.flatten()
				self.apply_mask(layer, layer_id, tensor)

	def prune(self, m, idx):
		if type(m) == torch.nn.Linear:
			tensor = m.weight.data
			tensor = tensor.flatten()
			tensor = torch.abs(tensor)
			prunable_features = torch.argsort(tensor).tolist()
			try:
				prunable_features = [i for i in prunable_features if i not in self.mask_indexes[idx]]
			except:
				pass
			prunable_features = prunable_features[:int(self.p*len(prunable_features))]
			if idx not in self.mask_indexes.keys():
				self.mask_indexes.update({idx: prunable_features})
			else:
				self.mask_indexes[idx] += prunable_features	

	def prune_weights(self):
		for layer_id, layer in enumerate(self.neural_network.modules()):
			self.prune(layer, layer_id)
		self.initialize_weights()
		self.apply_mask()

if __name__ == '__main__':
	model = ProjectModel()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
	pruner = PrunerModel(model, seed=0, p=0.0182)
	train(pruner.model, optimizer, 10)
	torch.save(pruner.model, "./models/original.pt")
	for pruning_iterations in range(5):
		print("Pruning for: p="+str((p*100)**(pruning_iterations+1)))
		pruner.prune_weights()
		train(pruner.model, optimizer, 10)
		torch.save(pruner.model, "./models/model"+str(pruning_iterations+1)+".pt")