import torch
from dataloader import MNIST_Dataset as ProjectDataset
from model import FullyConnectedModel as ProjectModel
from tqdm import tqdm

def train(model, optimizer, epochs):
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
			pred_y = model(batch_x)
			loss = criterion(pred_y, batch_y)
			running_loss += loss.item()
			pred = torch.argmax(pred_y, axis=1)
			acc = torch.mean((pred==batch_y).float())
			running_acc += acc.item()
			bar.set_description("VALIDATION:- "+str({"epoch": epoch+1, "loss": round(running_loss/(batch_idx+1), 4), "acc": round(running_acc/(batch_idx+1), 4)}))
		bar.close()

if __name__ == '__main__':
	model = ProjectModel()
	optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
	train(model, optimizer, 10)