# Enhancing Certified Robustness via Smoothed Weighted Ensembling

This is the code to replicate the results of our paper.

## Install dependencies

```python
conda create -n sween python=3.6
conda activate sween
conda install numpy matplotlib pandas seaborn
conda install scipy statsmodels
pip install setGPU
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

## Example

We will show how to train a SWEEN-3 model (see our paper for details) on CIFAR-10. Here we set $\sigma = 0.25$.

### Train candidate models for SWEEN

```python
python cand_train.py cifar10 resnet20 --name resnet_20 --sigma 0.25 
python cand_train.py cifar10 resnet26 --name resnet_26 --sigma 0.25 
python cand_train.py cifar10 resnet32 --name resnet_32 --sigma 0.25 
```

### Solving for ensemble weight

```python
python ens_train.py cifar10 3_model_ensemble --name resnet_20 resnet_26 resnet_32 --ens_name 3_model_ens --sigma 0.25
```

### Certify

Certification is done automatically during training. But if you want to certify the SWEEN model using our adpative prediction algorithm (see our paper), you need to do it manually using the following command:

```python
python ens_eval.py cifar10 3_model_ensemble --adp True --name resnet_20 resnet_26 resnet_32 --ens_ckpt 3_model_ens --ens_name adp_ens --sigma 0.25
```

### Visualize

You can use `analyze.py` to generate the certified accuracy plots of the SWEEN model and its candidate models.  To generate the plots used in our paper, simply run

```python
python analyze.py
```

This code reads from the logs that were generated when we certifiied our trained models, and automatically generates the plots presented in the paper.