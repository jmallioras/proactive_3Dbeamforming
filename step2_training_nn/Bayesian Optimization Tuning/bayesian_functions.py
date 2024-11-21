import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset
import torch
import time
import matplotlib.pyplot as plt
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.acquisition.monte_carlo import qExpectedImprovement
from lstm_functions import *
from gru_functions import *

def normalize(Y):
    min_y = np.min(Y)
    max_y = np.max(Y)
    return (Y-min_y)/(max_y-min_y)

def norm(x, x_min, x_max):
  return ((x-x_min)/(x_max-x_min))

def denorm(x,x_min,x_max):
  return np.add(np.multiply(x,(x_max-x_min)), x_min)


def find_iter(positions, maxiter):
  for x in range(maxiter):
    if not positions[x,-1,:].any():
      print(f"Left on iter: {x}")
      break
  return x

#Combines the Grid Search Parameters
def combos(x,y):
    combos = torch.zeros((len(x), len(y),2), dtype =int)
    for idx,nl in enumerate(x):
      combos[idx,:,0] = nl
    for idx,hs in enumerate(y):
      combos[:,idx,1] = hs
    return combos



def getLayers(input_size, HIDDEN_SIZES, output_size):
    layers = []
    layers.append(input_size)
    for h in range(HIDDEN_SIZES.shape[0]):
        layers.append(HIDDEN_SIZES[h].item())
    layers.append(output_size)
    return layers



def train_GRU_MTO(model, trainloader, testloader, tools, args):
    # Import necessary tools and parameters
    n_epochs, learning_rate, b_size, device = args
    optimizer, criterion = tools
    n_epochs_stop = 20
    epochs_no_improve = 0
    min_n_epochs = 300
    min_val_loss = 1
    early_stop = False
    TRAIN_RMSE = []
    TEST_RMSE = []
    # Start Training
    for epoch in range(n_epochs):
        start = time.time()
        # Training
        training_loss = 0.0
        with torch.set_grad_enabled(True):
            for i, (x, y) in enumerate(trainloader):
                # Forward pass
                output = model(x)  # Forward pass
                loss = torch.sqrt(criterion(output, y[:, -1, :2]))  #
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            # Update Train Error
            TRAIN_RMSE.append(training_loss / len(trainloader))

        # Evaluation
        evualtion_loss = 0.0
        with torch.no_grad():
            model.eval()
            for k, (xt, yt) in enumerate(testloader):
                output = model(xt)
                loss2 = torch.sqrt(criterion(output, yt[:,-1,:2]))
                evualtion_loss += loss2.item()
            TEST_RMSE.append(evualtion_loss / len(testloader))
            model.train()

        if (evualtion_loss / len(testloader)) < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = (evualtion_loss / len(testloader))
        else:
            epochs_no_improve += 1

        if epoch > min_n_epochs and epochs_no_improve == n_epochs_stop:
            print(f"Early stopping! @{epoch} EPOCHS\n")
            early_stop = True
            break

    # Define CRITERION
    criteria = np.sum(TEST_RMSE[epoch - 9:epoch + 1]) / 10 + 0.1 * max(0.0, np.subtract(
        np.sum(TEST_RMSE[epoch - 9:epoch + 1]) / 10, np.sum(TRAIN_RMSE[epoch - 9:epoch + 1]) / 10))
    print(f"Criterion: {criteria:.4f}, TRAIN/TEST RMSE: {TRAIN_RMSE[epoch]:.4f}/{TEST_RMSE[epoch]:.4f}")

    return criteria


def train_GRU_MTM(model, trainloader, testloader, tools, args):
    # Import necessary tools and parameters
    n_epochs, learning_rate, b_size, device = args
    optimizer, criterion = tools
    n_epochs_stop = 20
    epochs_no_improve = 0
    min_n_epochs = 300
    min_val_loss = 1
    early_stop = False
    TRAIN_RMSE = []
    TEST_RMSE = []
    # Start Training
    for epoch in range(n_epochs):
        start = time.time()
        # Training
        training_loss = 0.0
        with torch.set_grad_enabled(True):
            for i, (inputs, targets, lengths)  in enumerate(trainloader):
                inputs, targets= inputs.to(device), targets.to(device)
                lengths = lengths.to('cpu')
                # print(f"Inputs shape: {inputs.shape}")
                optimizer.zero_grad()
                outputs = model(inputs, lengths)

                mask = create_mask(lengths, targets.shape[1])
                loss = masked_loss(outputs, targets, mask, criterion)
                # Backward pass and optimization

                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            # Update Train Error
            TRAIN_RMSE.append(training_loss / len(trainloader))


        # Evaluation
        evaluation_loss = 0.0
        with torch.no_grad():
            model.eval()
            for k, (inputs, targets, lengths) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                lengths = lengths.to('cpu')
                outputs = model(inputs, lengths)

                mask = create_mask(lengths, targets.shape[1])
                loss2 = masked_loss(outputs, targets, mask, criterion)
                evaluation_loss += loss2.item()
            TEST_RMSE.append(evaluation_loss / len(testloader))
            model.train()

        if (evaluation_loss / len(testloader)) < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = (evaluation_loss / len(testloader))
        else:
            epochs_no_improve += 1

        if epoch > min_n_epochs and epochs_no_improve == n_epochs_stop:
            print(f"Early stopping! @{epoch} EPOCHS\n")
            early_stop = True
            break

    # Define CRITERION
    criteria = np.sum(TEST_RMSE[epoch - 9:epoch + 1]) / 10 + 0.1 * max(0.0, np.subtract(
        np.sum(TEST_RMSE[epoch - 9:epoch + 1]) / 10, np.sum(TRAIN_RMSE[epoch - 9:epoch + 1]) / 10))
    print(f"Criterion: {criteria:.4f}, TRAIN/TEST RMSE: {TRAIN_RMSE[epoch]:.4f}/{TEST_RMSE[epoch]:.4f}")

    return criteria

def target_function(tuning, args, datasets):
  result = []
  for x in tuning:
    print(f"------------TUNING:{x}------------")
    result.append(fitness_function(x, datasets,args))
  return torch.tensor(result)


def fitness_function(parameters, datasets, args):
    model_type, k_folds, n_epochs, learning_rate, batch_size, device = args
    # CHOOSE WHAT TYPE OF NEURAL NETWORK TO WORK ON
    # 0: LSTM-MTO or GRU-MTO
    # 1: LSTM-MTM or GRU-MTM

    # Unload parameters
    N_LAYERS, HIDDEN_SIZE, DROPOUT_RATE = parameters

    HIDDEN_SIZE = int(16 * HIDDEN_SIZE) # De-normalize hidden size

    # Convert dropout rate to [0,1] scale
    DROPOUT_RATE = round(0.1 * DROPOUT_RATE.item(), 1)
    print(f"Model Characteristics: LAYERS:{N_LAYERS}, SIZE:{HIDDEN_SIZE}, DROPOUT_RATE:{DROPOUT_RATE:.2f}\n")

    # K-FOLD
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Initialize model 
    '''
    if model_type == 0:
        model = LSTM_MTO_model(input_dim=2, hidden_dim=HIDDEN_SIZE, output_dim=2, num_layers=N_LAYERS, dropout = DROPOUT_RATE).to(device)
    elif model_type == 1:
        model = LSTM_MTM_model(input_dim=2, hidden_dim=HIDDEN_SIZE, output_dim=2, num_layers=N_LAYERS, dropout = DROPOUT_RATE).to(device)
    else:
        print("No model type selected")
    '''
    
    # Initialize model
    if model_type == 0:
        model = GRUModel_MTO(input_dim=2, hidden_dim=HIDDEN_SIZE, output_dim=2, num_layers=N_LAYERS, dropout = DROPOUT_RATE).to(device)
    elif model_type == 1:
        model = GRUModel_MTM(input_dim=2, hidden_dim=HIDDEN_SIZE, output_dim=2, num_layers=N_LAYERS, dropout = DROPOUT_RATE).to(device)
    else:
        print("No model type selected")
    
    # Initialize Dataloaders
    trainset, testset = datasets
    kfold_dataset = ConcatDataset([trainset, testset])

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Tools for training
    tools = (optimizer, criterion)
    args = (n_epochs, learning_rate, batch_size, device)

    # K-Fold Loop
    kfoldCriterion = torch.zeros(k_folds)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(kfold_dataset)):
        # kfold random index selection
        print(f"Fold {fold} begins..")
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = DataLoader(kfold_dataset, batch_size=batch_size, sampler=train_subsampler, drop_last=True)
        testloader = DataLoader(kfold_dataset, batch_size=batch_size, sampler=test_subsampler, drop_last=True)
        if model_type==0:
            kfoldCriterion[fold] = train_GRU_MTO(model, trainloader, testloader, tools, args)
        elif model_type==1:
            kfoldCriterion[fold] = train_GRU_MTM(model, trainloader, testloader, tools, args)
        print(f"Fold:{fold} ended with criterion: {kfoldCriterion[fold]:.4f}\n")
    fitness = torch.mean(kfoldCriterion)
    return -fitness



def generate_inital_data(n_samples, n_params, tools):
    bounds, args, datasets = tools
    tuning = torch.ones((n_samples, n_params), dtype=torch.int)
    for param in range(n_params):
        tuning[:, param] = torch.randint(bounds[0, param], bounds[1, param] + 1, (n_samples,), dtype=torch.int)

    exact_obj = target_function(tuning, args, datasets).unsqueeze(-1)
    best_observed_value = exact_obj.max().item()
    return tuning, exact_obj, best_observed_value


def get_next_points(PREVIOUS_X, PREVIOUS_Y, BEST_Y, bounds, n_points):
    single_model = SingleTaskGP(PREVIOUS_X, PREVIOUS_Y)
    mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
    fit_gpytorch_mll(mll)
    EI = qExpectedImprovement(model=single_model, best_f=BEST_Y)

    candidates, _ = optimize_acqf(acq_function=EI, bounds=bounds, q=n_points, num_restarts=200, raw_samples=512,
                                  options={"batch_limit": 5, "maxiter": 200})
    return candidates

