

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
python learn_accratio.py -Batch 64 -Ntherm 10 -Nsteps 10 -Nskip 4 -Nlayers 4 -Hs 10 -Ht 10 -target ring2d -epsilon 1.0 -alpha 0.0 -beta 1.0 -delta 1.0 -omega 1.0 -Nepoch 5000 

#Ising
python learn_accratio.py -Batch 64 -Ntherm 10 -Nsteps 10 -Nskip 10 -Nlayers 4 -Hs 40 -Ht 40 -target ising -K 0.44068679350977147 -L 4 -d 2 -epsilon 1.0 -alpha 0.0 -beta 1.0  -delta 1.0 -omega 1.0 -Nepoch 5000 
```

