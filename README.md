# Compositional PAC-Bayes

This is the official repo for "Compositional PAC-Bayes: Generalization of GNNs with persistence and beyond", published at NeurIPS 2024.


## Requirements
We use PyTorch and PyTorch Geometric. You also have install some auxiliary libraries, such as ```scipy```, ```sklearn```, ```ogb```, and ```gudhi```. 

## Usage

For instance, to run the model GIN with persistence, Gaussian point transformation with $q=100$, and SpecNorm regularization (regularization factor of 1e-7) on the dataset NCI1, we call
```
main.py --dataset NCI1 --q 100 --gnn gin --regularizer ours --alpha 1e-7 --point_transform gaussian
```

See the full list of arguments in the file ```main.py```. 

## Cite
```
@inproceedings{compositional-pac,
  title={Compositional PAC-Bayes: generalization of GNNs with persistence and beyond},
  author={Kirill Brilliantov and Amauri H. Souza and Vikas Garg},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
