

# Real NVP 

First, generate some training samples using metropolis

```python
python train/metropolis.py  -collectdata
```

It will write results to `data/ring2d_Nl8_Hs10_Ht10_mc.h5`. To see its content, do 

```python
h5ls -r data/ring2d_Nl8_Hs10_Ht10_mc.h5
```

in which `/results/samples         Dataset {1000, 16, 3}` stores the training data. (Nsamples, Bathsize, Nvars +1). 

Then, you can learn the probability either in the supervised  or unsupervised way. The supervised approach fits `model.logProbability(x)` to data. While the unsupervised way performs maximum log-likelihood estimation on the sample data.

```python
python train/learn_model.py -supervised -traindata data/ring2d_Nl8_Hs10_Ht10_mc.h5 
python train/learn_model.py -unsupervised -traindata data/ring2d_Nl8_Hs10_Ht10_mc.h5 
```

After learning, it will write results to disk, e.g. `data/ring2d_Nl8_Hs10_Ht10_sl.h5`

To inspect the hdf5 data, do 

```python
h5ls -r data/ring2d_Nl8_Hs10_Ht10_sl.h5
```

 To make plots, do 

```python
python plot/load_data.py -f data/ring2d_Nl8Hs10Ht10_sl.h5 -s
```

After learning, it will write the model to disk, e.g. the files in folder  `data/ring2d_Nl8_Hs10_Ht10_sl`

Next, one can use the real NVP net to generate new samples

```python
python train/sample_model.py -modelname data/ring2d_Nl8_Hs10_Ht10_sl/epoch490
```

Or, use the model to make MC update proposal

```python
python train/metropolis.py -modelname data/ring2d_Nl8_Hs10_Ht10_sl/epoch490 
```

