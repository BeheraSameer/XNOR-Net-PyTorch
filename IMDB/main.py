import torch
from torchtext import data
from torchtext import datasets
import random
import torch.nn as nn
import torch.nn.functional as F
import util
import torch.optim as optim
import models

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label.float())
        
        acc = binary_accuracy(predictions, batch.label.float())
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label.float())
            
            acc = binary_accuracy(predictions, batch.label.float())

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__=='__main__':

	# Select model here
	arch = "CNN"
	# arch = "Binary_CNN"

	# Training settings
	SEED = 1234

	torch.manual_seed(SEED)
	torch.cuda.manual_seed(SEED)
	torch.backends.cudnn.deterministic = True

	TEXT = data.Field(tokenize='spacy')
	LABEL = data.LabelField()

	train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

	train_data, valid_data = train_data.split(random_state=random.seed(SEED))

	TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d")
	LABEL.build_vocab(train_data)

	BATCH_SIZE = 64

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
	    (train_data, valid_data, test_data), 
	    batch_size=BATCH_SIZE, 
	    device=device)

	INPUT_DIM = len(TEXT.vocab)
	EMBEDDING_DIM = 100
	N_FILTERS = 100
	FILTER_SIZES = [3,4,5]
	OUTPUT_DIM = 1
	DROPOUT = 0.5

	if arch == "CNN":
		model = models.CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
	else:
		model = models.BinaryCNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
	
	pretrained_embeddings = TEXT.vocab.vectors

	model.embedding.weight.data.copy_(pretrained_embeddings)
	optimizer = optim.Adam(model.parameters())

	criterion = nn.BCEWithLogitsLoss()

	model = model.to(device)
	criterion = criterion.to(device)

	N_EPOCHS = 5

	for epoch in range(N_EPOCHS):

	    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
	    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
	    
	    print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

