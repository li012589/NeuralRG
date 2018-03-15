# To produce results in the paper

## Training 

```python
# Using MERA, L=8 Ising as example.
python learn_mera.py -Batch 1024 -Ntherm 5 -Nsteps 1 -Nskip 0 -Ndis 2 -Nlayers 8 -Hs 64 -Ht 64 -target ising -T 2.269185314213022 -L 8 -d 2 -delta 1.0 -omega 0.5 -Nepoch 5000 -lr 0.001 -exact 0.646769  -train_model 
```

or you can use the pre-trained demo provided.

```bash
mkdir -p data/learn_mera/
mv -f demo/* data/learn_mera/
```

## Check Training Results 

```python
# Plot loss and accept ratio curve.
python load_mcresults.py -f data/learn_mera/ising_L8_d2_T2.269185314213022_Nl8_Nd2_Hs64_Ht64_delta1.0_omega0.5_Batchsize1024_Ntherm5_Nsteps1_Nskips0_lr0.001_mc.h5 -exact -148.6550354919 -o loss_acc.pdf
```

## Plotting 

```python
# Plot renormalized variables.
python visualizeRG.py -h5file data/learn_mera/ising_L8_d2_T2.269185314213022_Nl8_Nd2_Hs64_Ht64_delta1.0_omega0.5_Batchsize1024_Ntherm5_Nsteps1_Nskips0_lr0.001_settings.h5 -modelname data/learn_mera/ising_L8_d2_T2.269185314213022_Nl8_Nd2_Hs64_Ht64_delta1.0_omega0.5_Batchsize1024_Ntherm5_Nsteps1_Nskips0_lr0.001/epoch3670 -s 

# Plot renormalized variables at various scales.
python visualizeRGscales.py -h5file data/learn_mera/ising_L8_d2_T2.269185314213022_Nl8_Nd2_Hs64_Ht64_delta1.0_omega0.5_Batchsize1024_Ntherm5_Nsteps1_Nskips0_lr0.001_settings.h5 -modelname data/learn_mera/ising_L8_d2_T2.269185314213022_Nl8_Nd2_Hs64_Ht64_delta1.0_omega0.5_Batchsize1024_Ntherm5_Nsteps1_Nskips0_lr0.001/epoch3670 -scale 0 -show
```

