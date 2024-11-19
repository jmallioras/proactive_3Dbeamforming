# Training Transformers and RNNs to forecast future DoAs

Using the dataset with DoA trajectories created in step 1, we intend to train and evaluate different types of neural networks (NNs) for the task in hand. 
- `train_model.py` is the main script that performs all necessary steps described below.

- `transformer_utils.py` contains the class definitions of different transformer (encoder-only) models that were tested
for this task. Finally, the model `contextDOAEncoder4` was used as the proposed architecture.

- `utility_functions.py` contains various important functions for data manipulation, plotting, and more that are used throughout
this script.

### Step 1: Data preparation
At this step, we inut the dataset with DoA trajectories containing the angles, angular velocities and angular
accelerations for each timestep. We then normalize the data based on their minimum and maximum values recorded
during the data collection process.

The normalization followed, is a bit unconvetional in the sense that we derive the min and max values from the dataset and
not by explicitly setting them for each input value. This is because the application is location-based and the angular 
sectors covered at each scenario may be different from the full [-90, 90] degrees of the operational range of the base
station. The DoA trajectories in the dataset cover a plethora of possible incoming signal directions, 
based on the predefined paths and the coverage map of the base station. Thus, by limiting the normalization sector
closer to the range of the recorded DoAs, we utilize the normalization sector better which improves training performance.

### Step 2: Configuring the NN designs
The 

### Step 3: Training and evaluating the NNs
