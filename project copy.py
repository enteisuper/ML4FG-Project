import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

bases={'A':0, 'C':1, 'G':2, 'T':3 } 
def one_hot(string):
    res = np.zeros( (4,len(string)), dtype=np.float32 )
    for j in range(len(string)):
        if string[j] in bases: # bases can be 'N' signifying missing: this corresponds to all 0 in the encoding
            res[ bases[ string[j] ], j ]=float(1.0)
    return(res)


import pickle
#genome = pickle.load(open(DATADIR+"hg38.pkl","rb")) # this is here in case there's hg38 data you want to analyse
genome = pickle.load(open("hg19.pickle","rb"))

class BedPeaksDataset(torch.utils.data.IterableDataset):
    def __init__(self, atac_data, genome, context_length):
        super(BedPeaksDataset, self).__init__()
        self.context_length = context_length
        self.atac_data = atac_data
        self.genome = genome
    def __iter__(self): 
        prev_end = 0
        prev_chrom = ""
        for i,row in enumerate(self.atac_data.itertuples()):
            midpoint = int(.5 * (row.start + row.end))
            seq = self.genome[row.chrom][ midpoint - self.context_length//2:midpoint + self.context_length//2]
            yield(one_hot(seq), np.float32(1)) # positive example

            if prev_chrom == row.chrom and prev_end < row.start: 
                midpoint = int(.5 * (prev_end + row.start))
                seq = self.genome[row.chrom][ midpoint - self.context_length//2:midpoint + self.context_length//2]
                yield(one_hot(seq), np.float32(0)) # negative example midway inbetween peaks, could randomize
            
            prev_chrom = row.chrom
            prev_end = row.end

import pandas as pd
binding_data = pd.read_csv("ENCFF300IYQ.bed.gz", sep='\t', usecols=range(6), names=("chrom","start","end","name","score","strand"))
binding_data = binding_data[ ~binding_data['chrom'].isin(["chrX","chrY"]) ] # only keep autosomes (non sex chromosomes)
binding_data = binding_data.sort_values(['chrom', 'start']).drop_duplicates() # sort so we can interleave negatives

test_chromosomes = ["chr1"]
test_data = binding_data[ binding_data['chrom'].isin( test_chromosomes ) ]

validation_chromosomes = ["chr2","chr3"]
validation_data = binding_data[ binding_data['chrom'].isin(validation_chromosomes) ]

train_chromosomes = ["chr%i" % i for i in range(4, 22+1)]
train_data = binding_data[ binding_data['chrom'].isin( train_chromosomes ) ]
train_dataset = BedPeaksDataset(train_data, genome, 100)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, num_workers = 0)


torch.manual_seed(2) # I played with different initialization here! 
my_first_conv_layer = nn.Conv1d(4, 1, 14, padding = 0)
def my_simplest_CNN(x): 
    net = my_first_conv_layer(x)
    net = net[:,0,:] # only one output channel! 
    # take maximum over channel ("global max pooling")
    net = torch.max(net, dim=1).values # max returns namedtuple (values, indices)
    net = torch.sigmoid(net) # aka logistic to get output in [0,1]
    return(net) 

x_np = one_hot("CCGCGNGGNGGCAG")
x_tensor = torch.tensor(x_np)
x_batch = x_tensor[None,:,:] # make a "batch" of size 1
filter_width = 5 
my_weights = torch.randn(1, 4, filter_width) # 1 output channels, 4 input channels, filter width = 5
convolution_output = F.conv1d(x_batch, my_weights)
print(my_simplest_CNN(x_batch))


# import timeit
# start_time = timeit.default_timer()
# torch.set_grad_enabled(True) # we'll need gradients
# for epoch in range(10): # run for this many epochs
#     losses = []
#     accuracies = []
#     for (x,y) in train_dataloader: # iterate over minibatches

#         output = my_simplest_CNN(x) # forward pass
#         # in practice (and below) we'll use more numerically stable built-in
#         # functions for the loss
#         loss = - torch.mean( y * torch.log(output) + (1.-y) * torch.log(1.-output) )
#         loss.backward() # back propagation

#         # iterate over parameter tensors: just the layer1 weights and bias here
#         for parameters in my_first_conv_layer.parameters(): 
#             parameters.data -= 1.0 * parameters.grad # in practive reduce or adapt learning rate
#             parameters.grad.data.zero_() # torch accumulates gradients so need to reset
        
#         losses.append(loss.detach().numpy()) # convert back to numpy
#         accuracy = torch.mean( ( (output > .5) == (y > .5) ).float() )
#         accuracies.append(accuracy.detach().numpy())  

#     elapsed = float(timeit.default_timer() - start_time)
#     print("Epoch %i %.2fs/epoch Loss: %.4f Acc: %.4f" % (epoch+1, elapsed/(epoch+1), np.mean(losses), np.mean(accuracies)))

validation_dataset = BedPeaksDataset(validation_data, genome, 100)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1000)
# accuracies = [ torch.mean( ( (my_simplest_CNN(x)  > .5) == (y > .5) ).float() ).detach().cpu().numpy() for (x,y) in validation_dataloader ]

class CNN_1d(nn.Module):
    def __init__(self, 
                 n_output_channels = 1, 
                 filter_widths = [15, 5], 
                 num_chunks = 5, 
                 max_pool_factor = 4, 
                 nchannels = [4, 32, 32],
                 n_hidden = 32, 
                 dropout = 0.2):
        
        super(CNN_1d, self).__init__()
        self.rf = 0 # running estimate of the receptive field
        self.chunk_size = 1 # running estimate of num basepairs corresponding to one position after convolutions

        conv_layers = []
        for i in range(len(nchannels)-1):
            conv_layers += [ nn.Conv1d(nchannels[i], nchannels[i+1], filter_widths[i], padding = 0),
                        nn.BatchNorm1d(nchannels[i+1]), # tends to help give faster convergence: https://arxiv.org/abs/1502.03167
                        nn.Dropout2d(dropout), # popular form of regularization: https://jmlr.org/papers/v15/srivastava14a.html
                        nn.MaxPool1d(max_pool_factor), 
                        nn.ELU(inplace=True)  ] # popular alternative to ReLU: https://arxiv.org/abs/1511.07289
            assert(filter_widths[i] % 2 == 1) # assume this
            self.rf += (filter_widths[i] - 1) * self.chunk_size
            self.chunk_size *= max_pool_factor

        # If you have a model with lots of layers, you can create a list first and 
        # then use the * operator to expand the list into positional arguments, like this:
        self.conv_net = nn.Sequential(*conv_layers)

        self.seq_len = num_chunks * self.chunk_size + self.rf # amount of sequence context required

        print("Receptive field:", self.rf, "Chunk size:", self.chunk_size, "Number chunks:", num_chunks)

        self.dense_net = nn.Sequential( nn.Linear(nchannels[-1] * num_chunks, n_hidden),
                                        nn.Dropout(dropout),
                                        nn.ELU(inplace=True), 
                                        nn.Linear(n_hidden, n_output_channels) )

    def forward(self, x):
        net = self.conv_net(x)
        net = net.view(net.size(0), -1)
        net = self.dense_net(net)
        return(net)

cnn_1d = CNN_1d()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_1d.to(device)
print("Input length:", cnn_1d.seq_len)

optimizer = torch.optim.Adam(cnn_1d.parameters(), amsgrad=True)
def run_one_epoch(train_flag, dataloader, cnn_1d, optimizer, device="cuda"):
    torch.set_grad_enabled(train_flag)
    cnn_1d.train() if train_flag else cnn_1d.eval() 

    losses = []
    accuracies = []

    for (x,y) in dataloader: # collection of tuples with iterator

        (x, y) = (x.to(device), y.to(device)) # transfer data to GPU

        output = cnn_1d(x) # forward pass
        output = output.squeeze() # remove spurious channel dimension
        loss = F.binary_cross_entropy_with_logits( output, y ) # numerically stable

        if train_flag: 
            loss.backward() # back propagation
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.detach().cpu().numpy())
        accuracy = torch.mean( ( (output > .5) == (y > .5) ).float() )
        accuracies.append(accuracy.detach().cpu().numpy())  
    
    return(np.mean(losses), np.mean(accuracies))

import timeit
train_accs = []
val_accs = []
patience = 10 # for early stopping
patience_counter = patience
best_val_loss = np.inf
check_point_filename = 'cnn_1d_checkpoint.pt' # to save the best model fit to date
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_1d.to(device)
for epoch in range(100):
    start_time = timeit.default_timer()
    train_loss, train_acc = run_one_epoch(True, train_dataloader, cnn_1d, optimizer, device)
    val_loss, val_acc = run_one_epoch(False, validation_dataloader, cnn_1d, optimizer, device)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    if val_loss < best_val_loss: 
        torch.save(cnn_1d.state_dict(), check_point_filename)
        best_val_loss = val_loss
        patience_counter = patience
    else: 
        patience_counter -= 1
        if patience_counter <= 0: 
            cnn_1d.load_state_dict(torch.load(check_point_filename)) # recover the best model so far
            break
    elapsed = float(timeit.default_timer() - start_time)
    print("Epoch %i took %.2fs. Train loss: %.4f acc: %.4f. Val loss: %.4f acc: %.4f. Patience left: %i" % 
          (epoch+1, elapsed, train_loss, train_acc, val_loss, val_acc, patience_counter ))

import analysis
print("entering analysis")
analysis.Analysis.analyze(validation_dataloader, device, cnn_1d)
