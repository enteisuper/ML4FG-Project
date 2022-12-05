# %%
import torch
import pickle
import bed_peaks_dataset
import pandas as pd
import cnn_1d
import matplotlib.pyplot as plt
import logomaker

binding_data = pd.read_csv("ENCFF300IYQ.bed.gz", sep='\t', usecols=range(6), names=("chrom","start","end","name","score","strand"))
binding_data = binding_data[ ~binding_data['chrom'].isin(["chrX","chrY"]) ] # only keep autosomes (non sex chromosomes)
binding_data = binding_data.sort_values(['chrom', 'start']).drop_duplicates() # sort so we can interleave negatives

#genome = pickle.load(open(DATADIR+"hg38.pkl","rb")) # this is here in case there's hg38 data you want to analyse
genome = pickle.load(open("hg19.pickle","rb"))

device = torch.device("cpu")

# creating saliency maps
test_chromosomes = ["chr1"]
test_data = binding_data[ binding_data['chrom'].isin( test_chromosomes ) ]

validation_chromosomes = ["chr2","chr3"]
validation_data = binding_data[ binding_data['chrom'].isin(validation_chromosomes) ]

train_chromosomes = ["chr%i" % i for i in range(4, 22+1)]
train_data = binding_data[ binding_data['chrom'].isin( train_chromosomes ) ]
# validation_dataset = bed_peaks_dataset.BedPeaksDataset(validation_data, genome, 100)
# validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1000)

model = cnn_1d.CNN_1d()
model.load_state_dict(torch.load("cnn_1d_checkpoint.pt"))

train_dataset = bed_peaks_dataset.BedPeaksDataset(train_data, genome, model.seq_len)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, num_workers = 0)
validation_dataset = bed_peaks_dataset.BedPeaksDataset(validation_data, genome, model.seq_len)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=1000)

torch.set_grad_enabled(False)
for (x_cpu,y_cpu) in validation_dataloader: 
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    output = model(x).squeeze()
    output = torch.sigmoid(output)
    delta_output = torch.zeros_like(x, device=device)
    # loop over all positions changing to each position nucleotide
    # note everything is implicitly parallelized over the batch here
    for seq_idx in range(model.seq_len): # iterate over sequence
        for nt_idx in range(4): # iterate over nucleotides
            x_prime = x.clone() # make a copy of x
            x_prime[:,:,seq_idx] = 0. # change the nucleotide to nt_idx
            x_prime[:,nt_idx,seq_idx] = 1.
            output_prime = model(x_prime).squeeze()
            output_prime = torch.sigmoid(output_prime)
            delta_output[:,nt_idx,seq_idx] = output_prime - output
    break # just do this for first batch
delta_output_np = delta_output.detach().cpu().numpy()
delta_output_np -= delta_output_np.mean(1, keepdims=True)
output_np = output.detach().cpu().numpy()
plt.figure(figsize = (12,12))
for i in range(1,5):
    ax = plt.subplot(4,1,i)
    pwm_df = pd.DataFrame(data = delta_output_np[i,:,:].transpose(), columns=("A","C","G","T"))
    crp_logo = logomaker.Logo(pwm_df, ax = ax) # CCGCGNGGNGGCAG or CTGCCNCCNCGCGG
    plt.title("True label: %i. Prob(y=1)=%.3f" % (y_cpu[i],output_np[i]))

plt.subplots_adjust(hspace = 0.3)

# %%
torch.set_grad_enabled(True)
x.requires_grad_() # tell torch we will want gradients wrt x (which we don't normally need)
output = model(x).squeeze()
output = torch.sigmoid(output)
dummy = torch.ones_like(output) # in a multiclass model this would be a one-hot encoding of y
output.backward(dummy) # to get derivative wrt to x
gradient_np = x.grad.detach().cpu().numpy()
output_np = output.detach().cpu().numpy()
saliency = gradient_np * x_cpu.detach().numpy()
plt.figure(figsize = (12,12))
for i in range(1,5):
    ax = plt.subplot(4,1,i) #,sharey=ax)
    pwm_df = pd.DataFrame(data = saliency[i,:,:].transpose(), columns=("A","C","G","T"))
    logomaker.Logo(pwm_df, ax=ax) # CCGCGNGGNGGCAG or CTGCCNCCNCGCGG
    plt.title("True label: %i. Prob(y=1)=%.3f" % (y_cpu[i],output_np[i]))

plt.subplots_adjust(hspace = 0.3)
# %%
