
**Contents**
- SSPdata/ -- contains mat files required to run some python scripts.
	- Shark_ssp.mat -- the original SW-06 dataset
	- ssp_mat_60.mat -- the SW-06 data interpolated to 60 points using shape preserving cubic splines
	- SSF_Hycom.mat -- the sound speed field from Hycom

- modules/ -- contains several python scripts required for all experiments
	- deep_models/ -- folder downloaded from https://github.com/DmitryUlyanov/deep-image-prior
	- cosine_annealing_with_warmup.py -- code for cosine annealing scheduler downloaded from  https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
	- deep_decoder.py -- convenience code downloaded from https://github.com/reinhardh/supplement_deep_decoder
	- deep_prior.py -- convenience code downloaded from https://github.com/DmitryUlyanov/deep-image-prior
	- losses.py -- implements losses required for training include L1, L2, and 2D TV
	- models.py -- implements multilayer perceptron, and multi-dimensional (1D, 2D, and 3D) U-Net
	- utils.py -- miscellaneous utilities
- requirements.txt -- contains a list of requirements to run the code base. Install the required packages using `pip install -r requirements.txt`

- SSP_decomp.py -- run this to derive two factor matrices. The scripts takes the SSP matrix as input and outputs two factor matrices. Suggest setting a high number of iterations and select the optimal number of iterations based on the rank estimation-iteration figure from the first training. Train again and take the basis function matrix and coefficient matrix obtained from the second training as the final output. In situations where the true value of training data cannot be obtained, the optimal number of iterations can still be obtained based on this method.

