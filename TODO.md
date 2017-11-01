- [ ] add GPU support for realNVP(if needed).

- [ ] add parallel between GPUs features to RealNVP.

- [x] a saver for RealNVP model.

- [ ] In _model/realNVP.py_, `saveModel` method of `RealNVP` class add check when add to save Dic

- [x] In _model/template.py_, mask can be handled more beautifully to better fit GPU.

- [ ] `Tensor.sum(dim=1)` is not right for Tensor shape more than 1D.

- [x] pytorch-lize `generate_sapmles.py`, in particule `test_prob(x)`

- [x] write a Metropolis MC for $\pi(x)$, i.e.`test_prob(x)` in `generate_samples.py`

- [x] use the trained real NVP net to make MC proposal 
      $$
      A(x\rightarrow x') = \min\left[ 1,  \frac{p(x)}{p(x')}\cdot \frac{\pi(x')}{\pi(x)}  \right],
      $$
      following these steps (this is actually called Metropolis Independent Sampler)

      ```
      z = Variable(torch.randn(batchsize, nvars), volatile=True)
      x = model.backward(z)
      r = (test_prob(x) - model.logp(x)) - (test_prob(xold)- model.logp(xold))
      ```


- [ ] make `metropolis.py` nicer: either use gaussian model or trained model for proposal 
- [ ] report acceptance ratio, autocorrelation time etc of the improved approach 
- [ ] bootstrap training, plot training results for each bootstrap iteration


