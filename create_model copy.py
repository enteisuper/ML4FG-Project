import torch
import pickle
import bed_peaks_dataset
import numpy as np
import timeit
import run_one_epoch
import pandas as pd
import cnn_1d
import train_model_batch_size

genome = pickle.load(open("hg19.pickle","rb"))
def train_model(cnn_1d, train_data, validation_data, epochs=20, patience=10, verbose = True):
    """
    Train a 1D CNN model and record accuracy metrics.
    """
    # Move the model to the GPU here to make it runs there, and set "device" as above
    # TODO CODE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_1d.to(device)

    # 1. Make new BedPeakDataset and DataLoader objects for both training and validation data.
    # TODO CODE
    genome = pickle.load(open("hg19.pickle","rb"))
    train_dataset = bed_peaks_dataset.BedPeaksDataset(train_data, genome, cnn_1d.seq_len)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, num_workers = 0)
    validation_dataset = bed_peaks_dataset.BedPeaksDataset(validation_data, genome, cnn_1d.seq_len)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1000)

    # 2. Instantiates an optimizer for the model. 
    # TODO CODE
    optimizer = torch.optim.Adam(cnn_1d.parameters(), amsgrad=True)

    # 3. Run the training loop with early stopping. 
    # TODO CODE
    train_accs = []
    val_accs = []
    patience_counter = patience
    best_val_loss = np.inf
    check_point_filename = 'cnn_1d_checkpoint.pt' # to save the best model fit to date
    for epoch in range(epochs):
        print(epoch)
        start_time = timeit.default_timer()
        train_loss, train_acc = run_one_epoch.run_one_epoch(True, train_dataloader, cnn_1d, optimizer, device)
        val_loss, val_acc = run_one_epoch.run_one_epoch(False, validation_dataloader, cnn_1d, optimizer, device)
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

    # 4. Return the fitted model (not strictly necessary since this happens "in place"), train and validation accuracies.
    return cnn_1d, train_accs, val_accs


binding_data = pd.read_csv("ENCFF300IYQ.bed.gz", sep='\t', usecols=range(6), names=("chrom","start","end","name","score","strand"))
binding_data = binding_data[ ~binding_data['chrom'].isin(["chrX","chrY"]) ] # only keep autosomes (non sex chromosomes)
binding_data = binding_data.sort_values(['chrom', 'start']).drop_duplicates() # sort so we can interleave negatives

test_chromosomes = ["chr1"]
test_data = binding_data[ binding_data['chrom'].isin( test_chromosomes ) ]
validation_chromosomes = ["chr2","chr3"]
validation_data = binding_data[ binding_data['chrom'].isin(validation_chromosomes) ]
train_chromosomes = ["chr%i" % i for i in range(4, 22+1)]
train_data = binding_data[ binding_data['chrom'].isin( train_chromosomes ) ]

my_cnn1d = cnn_1d.CNN_1d()
print(my_cnn1d.seq_len)
my_cnn1d
my_cnn1d, train_accs, val_accs = train_model(my_cnn1d, train_data, validation_data)

