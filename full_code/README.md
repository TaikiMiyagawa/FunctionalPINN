# Physics-informed Neural Networks for Functional Differential Equations: Cylindrical Approximation and Its Convergence Guarantees

## Important codes

- train.py: Training code.
- ./configs/config_train.py: Config file. Hyperparameter search space is given.
- ./dataprocesses/random_cp_dataset.py: Class RCPDv3 was used for training.
- ./losses/pdes.py: PDEFDs are defined (classes FTransportEquation and BurgersHopfEquation). Function softmax_coeffs is the loss-reweighting algorithm.
- models/pinns.py: PINN model is given here.
- utils/basis_functions.py: Orthogonal basis functions are defined here.
