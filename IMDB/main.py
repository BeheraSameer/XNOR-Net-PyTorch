import torch
from torchtext import data
from torchtext import datasets
import random
import torch.nn as nn
import torch.nn.functional as F
import util
import torch.optim as optim
import models
from tensorboardX import SummaryWriter

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
        
        # process the weights including binarization
        bin_op.binarization()
        
        predictions = model(batch.text).squeeze(1)
        
        loss = criterion(predictions, batch.label.float())
        
        acc = binary_accuracy(predictions, batch.label.float())
        
        loss.backward()

	# restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    bin_op.binarization()
    
    with torch.no_grad():
    
        for batch in iterator:

            predictions = model(batch.text).squeeze(1)
            
            loss = criterion(predictions, batch.label.float())
            
            acc = binary_accuracy(predictions, batch.label.float())

            epoch_loss += loss.item()
            epoch_acc += acc.item()
    bin_op.restore()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__=='__main__':

	# Select model here
	#arch = "CNN"
	arch = "Binary_CNN"

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
		model = models.Binary_CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
	# define the binarization operator
	bin_op = util.BinOp(model)
	# writer to build tensorboard model
	writer = SummaryWriter('runs/exp1/{}/'.format(arch)) 
	
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
	    writer.add_scalar('validation_loss', valid_loss, epoch)
	    writer.add_scalar('validation_accuracy',  100. * valid_acc, epoch)
	    bin_op.binarization()
	    #print(model.state_dict())
	    writer.add_histogram('/4_gram/weights',model.state_dict()['conv_1.conv.weight'], epoch)
	    writer.add_histogram('/4_gram/bias',model.state_dict()['conv_1.conv.bias'], epoch)
	    writer.add_histogram('/5_gram/weights',model.state_dict()['conv_2.conv.weight'], epoch)
	    writer.add_histogram('/5_gram/bias',model.state_dict()['conv_2.conv.bias'], epoch)
	    bin_op.restore()
	    
	    print('| Epoch: {:02} | Train Loss: {:.3f} | Train Acc: {:.2f}% | Val. Loss: {:.3f} | Val. Acc: {:.2f}% |'.format(epoch, train_loss, train_acc * 100, valid_loss, valid_acc*100))
	test_loss, test_acc = evaluate(model, test_iterator, criterion)

	print('| Test Loss: {:.3f} | Test Acc: {:.2f}% |'.format(test_loss, test_acc * 100)) 
