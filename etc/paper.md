# To produce results in the paper

## Training 

```python
# Using MERA, L=8 Ising as example.
python learn_mera.py -Batch 1024 -Ntherm 5 -Nsteps 1 -Nskip 0 -Ndis 1 -Nlayers 8 -Hs 64 -Ht 64 -target ising -T 2.269185314213022 -L 8 -d 2 -epsilon 0.0 -beta 1.0  -delta 1.0 -omega 0.5 -Nepoch 5000 -lr 0.001 -exact 0.646769  -train_model 
```

or you can use the pre-trained demo provided.

```bash
mkdir -p data/learn_mera/
mv demo/* data/learn_mera/
```

## Check Training Results 

```python
# Sample using the trained model.
python learn_accratio.py -Batch 64 -Ntherm 5 -Nsteps 1 -Nskip 0 -Nlayers 10 -Hs 4 -Ht 4 -target ising -K 0.44068679350977147 -L 4 -d 2 -epsilon 1.0 -beta 1.0  -delta 1.0 -omega 0.0  -Nepoch 5000 -lr 0.01 -exact 0.371232 -modelname data/learn_acc/ising_L8_d2_K0.44068679350977147_Nl10_Hs4_Ht4_epsilon1.0_beta1.0_delta1.0_omega0.0_Batchsize64_Ntherm5_Nsteps1_Nskips0_lr0.01/epoch230

# Plot proposed and accepted configurations.
python plot_configs.py -f data/learn_acc/ising_L8_d2_K0.44068679350977147_Nl10_Hs4_Ht4_epsilon1.0_beta1.0_delta1.0_omega0.0_Batchsize64_Ntherm5_Nsteps1_Nskips0_lr0.01_mc.h5 

# Plot loss and accept ratio curve.
python load_mcresults.py -f data/learn_mera/ising_L8_d2_T2.269185314213022_Nl8_Nd2_Hs64_Ht64_epsilon0.0_beta1.0_delta1.0_omega0.5_Batchsize1024_Ntherm5_Nsteps1_Nskips0_lr0.001_mc.h5 -exact -148.6550354919 -o loss_acc.pdf 
```

## Plotting 

```python
# Plot renormalized variables.
python visualizeRG.py -h5file data/learn_mera/ising_L8_d2_T2.269185314213022_Nl8_Nd2_Hs64_Ht64_epsilon0.0_beta1.0_delta1.0_omega0.5_Batchsize1024_Ntherm5_Nsteps1_Nskips0_lr0.001_settings.h5  -modelname data/learn_mera/ising_L8_d2_T2.269185314213022_Nl8_Nd2_Hs64_Ht64_epsilon0.0_beta1.0_delta1.0_omega0.5_Batchsize1024_Ntherm5_Nsteps1_Nskips0_lr0.001/epoch1990 -s        

# Plot renormalized variables at various scales.
python visualizeRGscales.py -h5file data/learn_mera/ising_L8_d2_T2.269185314213022_Nl8_Nd2_Hs64_Ht64_epsilon0.0_beta1.0_delta1.0_omega0.5_Batchsize1024_Ntherm5_Nsteps1_Nskips0_lr0.001_settings.h5  -modelname data/learn_mera/ising_L8_d2_T2.269185314213022_Nl8_Nd2_Hs64_Ht64_epsilon0.0_beta1.0_delta1.0_omega0.5_Batchsize1024_Ntherm5_Nsteps1_Nskips0_lr0.001/epoch1990  -scale 0 -o scale0.pdf

# Make animation.
convert -background white  -delay 20 -dispose previous  $(ls data/learn_mera/ising_L8_d2_T2.269185314213022_Nl8_Nd2_Hs64_Ht64_epsilon0.0_beta1.0_delta1.0_omega0.5_Batchsize1024_Ntherm5_Nsteps1_Nskips0_lr0.001/epoch*.png|sort -V| head -n 50)  animation.gif
```

