

# Real NVP 



## Main scheme

`sample_model.py` will generate data for training,

`learn_model.py` will learn a RealNVP model, which can be used to speed up metropolis in return. 

In the bootstrap training, these two scripts exchange **data** and **model** and improve iteratively.   

## How to Run 

First, generate some training samples using metropolis

```bash
python ./sampler.py -target ring2d -collectdata
```

It will write results to `data/ring2d_Nl8_Hs10_Ht10_mc.h5`. `-collectdata` tells it to collect training data.  To see its content, do 

```bash
h5ls -r data/ring2d_Nl8_Hs10_Ht10_mc.h5
```

in which `/results/samples         Dataset {1000, 16, 3}` stores the training data. (Nsamples, Bathsize, Nvars +1). 

Then, you can learn the probability either in the supervised or unsupervised way. The supervised approach fits `model.logProbability(x)` to data. While the unsupervised way performs maximum log-likelihood estimation on the sample data.

```python
# supervised
python ./learn_model.py -target ring2d -supervised -traindata data/ring2d_Nl8_Hs10_Ht10_mc.h5 
# unsupervised
python ./learn_model.py -target ring2d -unsupervised -traindata data/ring2d_Nl8_Hs10_Ht10_mc.h5 
```

After learning, it will write results and model to disk, e.g. 

```bash
data/ring2d_Nl8_Hs10_Ht10_sl.h5
data/ring2d_Nl8_Hs10_Ht10_sl/epoch0
data/ring2d_Nl8_Hs10_Ht10_sl/epoch10
...
data/ring2d_Nl8_Hs10_Ht10_sl/epoch490
```

The `.h5` file contains the results of the model, while the folder contains the trained model at each epoch. 

To inspect the hdf5 data, do 

```bash
h5ls -r data/ring2d_Nl8_Hs10_Ht10_sl.h5
```

 To make plots, do 

```bash
python analysis/load_data.py -f data/ring2d_Nl8_Hs10_Ht10_sl.h5 -s
```

Next, one can use the real NVP net to generate new samples

```bash
python ./sample_model.py -modelname data/ring2d_Nl8_Hs10_Ht10_sl/epoch490
```

Or, use the model to make MC update proposal

```bash
python ./sampler.py -target ring2d -modelname data/ring2d_Nl8_Hs10_Ht10_sl/epoch490 
```

By providing `-collectdata` to the command, one will get new train data. Which can be used to improve the model. 

Run it for Ising model:

```python
python sampler.py -target ising -Nsamples 1000  -Nskip 1 -Batchsize 10 -collectdata  -K 0.44068679350977147  -L 4 -d 2  -sampler hmc -interSteps 100 -stepSize 0.1 
```

Learn by maximazing the acceptance rate
```python
#ring2D
python learn_accratio.py -Batch 64 -Ntherm 10 -Nsteps 10 -Nskip 10 -Nlayers 4 -Hs 10 -Ht 10 -target ring2d -epsilon 1.0 -alpha 0.0 -beta 1.0 -delta 1.0 -omega 1.0 -Nepoch 5000 

#Ising
python learn_accratio.py -Batch 64 -Ntherm 5 -Nsteps 1 -Nskip 0 -Nlayers 10 -Hs 4 -Ht 4 -target ising -T 2.5 -L 8 -d 2 -epsilon 1.0 -beta 1.0  -delta 1.0 -omega 0.0  -Nepoch 5000 -lr 0.001 -exact 0.177921 -train_model 

#or 
python learn_accratio.py -Batch 256 -Ntherm 0 -Nsteps 5 -Nskip 0 -Nlayers 8 -Hs 16 -Ht 16 -target ising -T 2.5 -L 8 -d 2 -epsilon 0.0 -beta 1.0  -delta 0.0 -omega 1.0  -Nepoch 1000 -lr 0.001 -exact 0.177921 -train_model
```

To check the results 

```python
#sample using the model
python learn_accratio.py -Batch 64 -Ntherm 5 -Nsteps 1 -Nskip 0 -Nlayers 10 -Hs 4 -Ht 4 -target ising -K 0.44068679350977147 -L 4 -d 2 -epsilon 1.0 -beta 1.0  -delta 1.0 -omega 0.0  -Nepoch 5000 -lr 0.01 -exact 0.371232 -modelname data/learn_acc/ising_L4_d2_K0.44068679350977147_Nl10_Hs4_Ht4_epsilon1.0_beta1.0_delta1.0_omega0.0_Batchsize64_Ntherm5_Nsteps1_Nskips0_lr0.01/epoch230

# plot proposed and accepted configurations
python plot_configs.py -f data/learn_acc/ising_L4_d2_K0.44068679350977147_Nl10_Hs4_Ht4_epsilon1.0_beta1.0_delta1.0_omega0.0_Batchsize64_Ntherm5_Nsteps1_Nskips0_lr0.01_mc.h5 

```





### Exact Ising results 

| $d=2,T=2.5$ | PBC                   | OBC                  |
| :---------: | --------------------- | -------------------- |
|    L=16     | 0.138871+/-0.000273   | 0.130933+/-0.000262  |
|    L=64     | 0.0344761+/-0.0001465 | 0.0341328+/-0.000158 |
|             |                       |                      |



| $d=2,T=T_c$ |          PBC          |          OBC          |
| :---------: | :-------------------: | :-------------------: |
|    $L=2$    |                       | 0.56882 +/- 0.000409  |
|    $L=4$    | 0.761761 +/- 0.000421 | 0.371232 +/- 0.000429 |
|    $L=6$    | 0.693504 +/- 0.000436 | 0.30321 +/- 0.000397  |
|    $L=8$    | 0.646769 +/- 0.000445 | 0.266751 +/- 0.00038  |

| $d=2,T=2.5$ | PBC  |          OBC           |
| :---------: | :--: | :--------------------: |
|    $L=4$    |      |   0.309222+/-0.0004    |
|    $L=8$    |      | 0.177921 +/- 0.000304  |
|   $L=16$    |      | 0.0905703 +/- 0.000219 |






