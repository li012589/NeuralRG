### V 0.2

1. Vars naming changes: Nvars -> numVars, etc.
2. create templates for layers and realNVP for flexibile use in future.
3. use created templates re-write realnvp.py
4. use new class for NN instead of list of torch.nn.*, so that we can save
  trained NN.