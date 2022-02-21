import pandas as pd
import numpy as np
import torch

class MNIST_Dataset(torch.utils.data.Dataset):
	def __init__(self, mode, np=1, shuffle=False, path_train="./data/mnist_train.csv", path_val="./data/mnist_test.csv"):
		self.mode = mode
		self.path_val = path_val
		self.path_train = path_train
		self.np = np
		self.data = self.read_data()
		self.data = self.data[:int(len(self.data)*self.np)]

	def read_data(self):
		if self.mode == "train":
			path = self.path_train
		else:
			path = self.path_val
		df = pd.read_csv(path)
		df = df.values
		return df

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return torch.tensor(self.data[idx][1:]), torch.tensor(self.data[idx][0])

if __name__ == '__main__':
	training_data = MNIST_Dataset(mode="train")
	train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
	for batch_x, batch_y in train_dataloader:
		print(batch_x.shape, batch_y.shape)