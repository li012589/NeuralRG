# Real NVP 

First, generate some samples for training 

```python
python generate_samples.py > train.dat
```

It will write three column data like this, where the last column is the log-probability

```
#x, logp(x) 
-0.307216382173 -1.8566450028 -1.19416949971 
-3.38369414161 3.54963339913 -2.8692304703 
0.6690070947 -2.29146205847 -0.769982650559 
-2.14597762639 1.06654130027 -1.44260068878
... 
```

Then, you can learn the probability either in the supervised or unsupervised way. The supervised approach fits `model.logp(x)` to data. While the unsupervised way performs maximum log-likelihood estimation on the sample data.

```python
python sl_realnvp.py 
python ul_realnvp.py
```

After learning, one can use the real NVP net to generate new samples by doing

```python
z = Variable(torch.randn(Nsamples, Nvars))
x = model.backward(z)
```

The log-probability of which is `model.logp(x)`
