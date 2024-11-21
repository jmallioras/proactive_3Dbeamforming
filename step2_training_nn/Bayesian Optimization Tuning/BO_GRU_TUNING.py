import torch
import time
from bayesian_functions import *
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.io import loadmat
from dataset_functions import *
from lstm_functions import *
import os
import csv
from torch.utils.data import random_split, DataLoader, TensorDataset

# Device configuration
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")


#Initialize NN model to be used:
# CHOOSE WHAT TYPE OF NEURAL NETWORK TO WORK ON
  # 0: GRU-MTO
  # 1: GRU-MTM
mt = ["GRU-MTO", "GRU-MTM"]
MODEL_TYPE = 1
DATASET_SIZE = 100000
print(f"BAYESIAN OPTIMIZATION FOR TUNING: {mt[MODEL_TYPE]}")

pwd = os.getcwd()

RESULTS_PATH = pwd+'/BAYESIAN_GRU_ΜΤM/'
IM_PATH = RESULTS_PATH+'/FIGURES'

try:
  os.mkdir(RESULTS_PATH)
except:
  print("Directory ready")

try:
  os.mkdir(IM_PATH)
except:
  print("Directory ready")


# Dataset preparation
pwd = os.getcwd()
window_size = 5
step_ahead = 1
data = loadmat(pwd + '/Dataset/total_dataset.mat')
block_dataset = data['out']
print(f"Imported Dataset of shape:[{block_dataset.shape[0]} x {block_dataset.shape[1]} x {block_dataset.shape[2]}]")

# Convert to tensor
block_dataset = torch.from_numpy(block_dataset).float()

# Keep a smaller part for optimization
block_dataset = block_dataset[torch.randperm(block_dataset.shape[0])] # Shuffle rows
block_dataset = block_dataset[:DATASET_SIZE]

# Normalize dataset to [-1, 1]
# Angles
denorm_az = torch.tensor([torch.min(block_dataset[:,:,0]), torch.max(block_dataset[:,:,0])])
denorm_el = torch.tensor([torch.min(block_dataset[:,:,1]), torch.max(block_dataset[:,:,1])])

block_dataset[:,:,0] = normalize_data(block_dataset[:,:,0], torch.min(block_dataset[:,:,0]), torch.max(block_dataset[:,:,0]))
block_dataset[:,:,1] = normalize_data(block_dataset[:,:,1], torch.min(block_dataset[:,:,1]), torch.max(block_dataset[:,:,1]))

# Keep Angles only
block_dataset = block_dataset[:,:,:2]

if MODEL_TYPE == 0:
  trainset = block_dataset[:int(0.9*len(block_dataset))]
  valset = block_dataset[int(0.9*len(block_dataset)):]
  # Shuffle the rows of each dataset
  trainset = trainset[torch.randperm(trainset.shape[0])]
  valset = valset[torch.randperm(valset.shape[0])]
  train_dataset = DOADataset(trainset,device)
  val_dataset = DOADataset(valset,device)
elif MODEL_TYPE==1:
  # Random masking of input
  masked_inputs, masked_targets, lengths = random_masking(block_dataset)
  # Create a tensor dataset
  dataset = TensorDataset(masked_inputs, masked_targets, lengths)
  # Define the size of the test set
  val_size = int(0.1 * len(dataset))  # 20% for testing
  train_size = len(dataset) - val_size  # 80% for training
  # Split the dataset
  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
else:
  print("Wrong model type")

# Set the batch size
batch_size = 64

# Create DataLoaders
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# NN and Training Specifics
KFOLDS = 3
EPOCHS = 200
lr = 0.0001

# BO CHARACTERISTICS
INITIAL_SAMPLES = 10
UNKNOWNS = 3
NUMBER_OF_RUNS = 30

# DEFINE BOUNDS
BOUNDS = torch.tensor([[1,8,0], [4,64,4]],dtype=torch.int)

# Initialize Arrays
PREVIOUS_SETS = torch.tensor([])
PREVIOUS_CRITERIA = torch.tensor([])
BEST_PROGRESSION = []

# Prepare Dataset for specific model case
datasets = (train_dataset, val_dataset)

# TOOLS
fitnessF_args = MODEL_TYPE, KFOLDS, EPOCHS, lr, batch_size, device
TOOLS = BOUNDS, fitnessF_args, datasets

bo_start = time.time()


'''
try:
  PREVIOUS_SETS = torch.load(RESULTS_PATH+"/SETS.pt")
  PREVIOUS_CRITERIA =  torch.load(RESULTS_PATH+"/CRITERIA.pt")
  BEST_PROGRESSION = torch.load(RESULTS_PATH+"/BEST_FITNESS.pt")
  iter = len(BEST_PROGRESSION)
  print(f"Previous Optimization Data Loaded (Left on iteration: {iter})")
except:
  print(f"PREPARING {INITIAL_SAMPLES} NEW SAMPLES")
  # GENERATE INITIAL DATA
  INITIAL_SETS, INITIAL_CRITERIA, BEST_INIT_CRITERION = generate_inital_data(INITIAL_SAMPLES, UNKNOWNS, TOOLS)
  # SAVE INITIAL DATA
  PREVIOUS_SETS = torch.cat([PREVIOUS_SETS,INITIAL_SETS])
  PREVIOUS_CRITERIA = torch.cat([PREVIOUS_CRITERIA,INITIAL_CRITERIA])
  BEST_PROGRESSION.append(BEST_INIT_CRITERION)
  iter=0
'''

print(f"PREPARING {INITIAL_SAMPLES} NEW SAMPLES")
# GENERATE INITIAL DATA
INITIAL_SETS, INITIAL_CRITERIA, BEST_INIT_CRITERION = generate_inital_data(INITIAL_SAMPLES, UNKNOWNS, TOOLS)
# SAVE INITIAL DATA
PREVIOUS_SETS = torch.cat([PREVIOUS_SETS,INITIAL_SETS])
PREVIOUS_CRITERIA = torch.cat([PREVIOUS_CRITERIA,INITIAL_CRITERIA])
BEST_PROGRESSION.append(BEST_INIT_CRITERION)
iter=0


print("------------ INITIAL SETS EXTRACTED ------------\n")
print("------------------------------------------------\n")
print("             STARTING BAYESIAN OPT              \n")


# START BAYESIAN OPTIMIZATION
for ITERATION in range(NUMBER_OF_RUNS):
  print("------------------------------------------------\n")
  print(f"Run: {ITERATION+iter}")
  new_candidates = get_next_points(PREVIOUS_SETS.float(), PREVIOUS_CRITERIA, BEST_PROGRESSION[-1], BOUNDS.float(), 1).int()
  print(f"New Candidates are: {new_candidates}")
  new_results = target_function(new_candidates, fitnessF_args, datasets).unsqueeze(-1)

  PREVIOUS_SETS = torch.cat([PREVIOUS_SETS, new_candidates])
  PREVIOUS_CRITERIA = torch.cat([PREVIOUS_CRITERIA, new_results])
  BEST_PROGRESSION.append(PREVIOUS_CRITERIA.max().item())
  print(f"Best point performs this way: {BEST_PROGRESSION[-1]}")

  # Save progress
  torch.save(PREVIOUS_SETS, RESULTS_PATH+"SETS.pt")
  torch.save(PREVIOUS_CRITERIA, RESULTS_PATH+"CRITERIA.pt")
  torch.save(BEST_PROGRESSION, RESULTS_PATH+ "BEST_FITNESS.pt")

  # Plot Progression
  font = fm.FontProperties(fname='Times New Roman')
  plt.figure(dpi=600)
  plt.plot(BEST_PROGRESSION ,color='m', label = 'Best Criterion', linewidth = 2)
  plt.xlabel("Iteration",fontsize=16)
  plt.ylabel("Criterion",fontsize=16)
  plt.title('BO Objective Function progression', fontsize=20)
  plt.savefig(RESULTS_PATH+f"iter_{ITERATION}.pdf",bbox_inches='tight')
  plt.close()

for i in range(PREVIOUS_SETS.shape[0]):
    print(f"Set: {PREVIOUS_SETS[i,:]}, CRITERION: {PREVIOUS_CRITERIA[i]}")

print(f"BO FINISHED, TIME ELAPSED: {((time.time()-bo_start)/60)/60}hrs")