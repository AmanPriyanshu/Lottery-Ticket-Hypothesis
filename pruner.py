import torch
from dataloader import MNIST_Dataset as ProjectDataset
from model import FullyConnectedModel as ProjectModel
from tqdm import tqdm

def train(pruner, model, optimizer, epochs, base=False):
	training_data = ProjectDataset(mode="train")
	train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64)
	validation_data = ProjectDataset(mode="val")
	validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=64)
	criterion = torch.nn.CrossEntropyLoss()

	for epoch in range(epochs):
		model.train()
		running_loss, running_acc = 0, 0
		bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
		for batch_idx, (batch_x, batch_y) in bar:
			batch_x = batch_x.float()
			optimizer.zero_grad()
			if not base:
				pruner.apply_mask()
			pred_y = model(batch_x)
			loss = criterion(pred_y, batch_y)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			pred = torch.argmax(pred_y, axis=1)
			acc = torch.mean((pred==batch_y).float())
			running_acc += acc.item()
			bar.set_description("TRAINING:- "+str({"epoch": epoch+1, "loss": round(running_loss/(batch_idx+1), 4), "acc": round(running_acc/(batch_idx+1), 4)}))
		bar.close()
		model.eval()
		running_loss, running_acc = 0, 0
		bar = tqdm(enumerate(validation_dataloader), total=len(validation_dataloader))
		for batch_idx, (batch_x, batch_y) in bar:
			batch_x = batch_x.float()
			if not base:
				pruner.apply_mask()
			pred_y = model(batch_x)
			loss = criterion(pred_y, batch_y)
			running_loss += loss.item()
			pred = torch.argmax(pred_y, axis=1)
			acc = torch.mean((pred==batch_y).float())
			running_acc += acc.item()
			bar.set_description("VALIDATION:- "+str({"epoch": epoch+1, "loss": round(running_loss/(batch_idx+1), 4), "acc": round(running_acc/(batch_idx+1), 4)}))
		bar.close()

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
				self.mask(layer, layer_id, tensor)

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
	pruner = ModelAugmenter(model, seed=0, p=0.0182)
	train(pruner, pruner.neural_network, optimizer, 3, base=True)
	torch.save(pruner.neural_network, "./models/original.pt")
	for pruning_iterations in range(5):
		print("Pruning for: p="+str((pruner.p*100)**(pruning_iterations+1)))
		pruner.prune_weights()
		torch.save(pruner.neural_network, "./models/model_pruned"+str(pruning_iterations+1)+".pt")
		train(pruner, pruner.neural_network, optimizer, 3)
		torch.save(pruner.neural_network, "./models/model"+str(pruning_iterations+1)+".pt")