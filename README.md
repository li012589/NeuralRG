

# Real NVP 

First, generate some training samples using metropolis

```python
python train/metropolis.py -Nvars 2 -Nlayers 4 -Hs 10 -Ht 10 > train.dat
```

It will write three column data like this, where the last column is the log-probability

```
1.41873 -2.37144 -1.8213218450546265
-0.135188 -1.87742 -0.04330186918377876
-0.940416 1.6424 -0.0360623337328434
-1.18222 1.79976 -0.07345814257860184
... 
#accratio: 0.26450299999999755
```

Then, you can learn the probability either in the supervised  or unsupervised way. The supervised approach fits `model.logProbability(x)` to data. While the unsupervised way performs maximum log-likelihood estimation on the sample data.

```python
python train/learn_model.py -supervised -Nlayers 4 -Hs 10 -Ht 10
python train/learn_model.py -unsupervised -Nlayers 4 -Hs 10 -Ht 10
```

After learning, it will write the model to disk, e.g. the file `Nvars2Nlayers4Hs10Ht10.realnvp`

Next, one can use the real NVP net to generate new samples

```python
python train/sample_model.py -Nvars 2 -Nlayers 4 -Hs 10 -Ht 10
```

Or, use the model to make MC update proposal

```python
python train/metropolis.py -Nvars 2 -Nlayers 4 -Hs 10 -Ht 10 -loadmodel > test.dat
```

