- [ ] add GPU support for realNVP(if needed).
- [ ] add parallel between GPUs features to RealNVP.
- [ ] a saver for RealNVP model.

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

