

# NeuralRG 

Pytorch implement of arXiv paper: Shuo-Hui Li and Lei Wang, *Neural Network Renormalization Group* [arXiv:1802.02840](https://arxiv.org/abs/1802.02840).

**NeuralRG** is a deep generative model using variational renormalization group approach, it's also a kind of [normalizing flow](https://blog.evjang.com/2018/01/nf1.html), it is composed by layers of bijectors (In our implementation, we use [realNVP](https://arxiv.org/abs/1605.08803)). After training, it can generate statistically independent physical configurations with tractable likelihood via directly sampling.

## How does NeuralRG work

In NeuralRG Network(a), we use realNVP (b) networks as building blocks, realNVP is a kind of bijectors(a normalizing flow), they can transform one distribution into other distribution and revert this process. For multi-in-multi-out blocks, we call they disentanglers(gray blocks in (a)), and for multi-in-single-out blocks, we can they decimators(white blocks in (a)). And stacking multiple layers of these blocks into a hierarchical structure forms NeuralRG network, so NeuralRG is also a bijector. In inference process, each layer tries to "separate" entangled variables into independent variables, and at layers composed of decimators, we only keep one of these independent variables, this is renormalization group.

![NeuralRG Network](etc/Nflow.png)

The structure we used to construct realNVP networks into NeuralRG network is inspired by multi-scale entanglement renormalization ansatz (MERA), as shown in (a). Also, the process of variable going through our network can be viewed as a renormalization process.

The resulted effect of a trained NeuralRG network can be visualized using gradients plot (a) and MI plot of variables of the same layer (b)(c). The latent variables of NeuralRG appears to be a nonlinear and adaptive generalization of wavelet basis.

![gradientAndMi](etc/gradAndMi.png)

## How to Run 

### Train

Use `main.py` to train model. Options available are:

* `folder` specifies saving path. At that path a `parameters.hdf5` will be created to keep training parameters, a `pic` folder will be created to keep plots, a `records` folder will be created to keep saved HMC records, and a `savings` folder to save models in.
* `name` specifies model's name. If not specified, a name will be created according to training parameters.
* `epochs`, `batch`, `lr`, `savePeriod` are the number of training epochs, batch size, learning rate, the number of epochs before saving.
* `cuda` indicates on which GPU card should the model be trained, the default value is -1, which means running on CPU.
* `double` indicates if use double float.
* `load` indicates if load a pre-trained model. If true, will try to find a pre-trained model at where `folder` suggests. Note that if true all other parameters will be overwritten with what saved in `folder`'s `parameters.hdf5`.
* `nmlp`, `nhidden` are used to construct MLP networks inside of realNVP networks. `nmlp` is the number of layers in MLP networks and `nhidden` is the number of hidden neurons in each layer.
* `nlayers` is used to construct realNVP networks, it suggests how many layers in each realNVP networks.
* `nrepeat` is used to construct MERA network, it suggests how many layers of bijectors inside of each layer of MERA network.
* `L`, `d`, `T` are used to construct the Ising model to learn, `L` is the size of configuration, `d` is the dimension, and `T` is the temperature.

For example, to train the Ising model mentioned in our paper:

```bash
python ./main.py -epochs 5000 -folder ./opt/16Ising -batch 512 -nlayers 10 -nmlp 1 -nhidden 10 -L 16 -nrepeat 1 -savePeriod 100
```



### Plot

Use `plot.py` to plot the loss curve and HMC result results. Options available are:

* `folder` specifies saving path. `plot.py` will use the data saved in that path to plot. And if `save` is true, the plot will be saved in `folder`'s `pic` folder.
* `per` specifies how many seconds between each refresh.
* `show`, `save` specifies if will show/save the plot.
* `exact` specifies the exact result of HMC.

For example, to plot along with the training process mentioned above:

```bash
python ./plot.py -folder ./opt/16Ising2 -per 30 -exact 0.544445
```



## Citation

If you use this code for your research, please cite our [paper](https://arxiv.org/abs/1802.02840):

```
@article{neuralRG,
  Author = {Shuo-Hui Li and Lei Wang},
  Title = {Neural Network Renormalization Group},
  Year = {2018},
  Eprint = {arXiv:1802.02840},
}
```

## Contact

For questions and suggestions, contact Shuo-Hui Li at [contact_lish@iphy.ac.cn](mailto:contact_lish@iphy.ac.cn).



## ETC

### Exact Z for Ising n = 16, T from 2.0 to 2.5

| T                 | $lnZ$              | fix                | sum/n      |
| ----------------- | ------------------ | ------------------ | ---------- |
| 2.0               | 263.29621043402125 | 399.418793393396   | 2.58873048 |
| 2.1               | 252.8952741554581  | 381.48763983688104 | 2.47805826 |
| 2.2               | 243.9756979903575  | 364.9500280838529  | 2.37861612 |
| 2.269185314213022 | 238.64225663513287 | 354.2334832224499  | 2.31592086 |
| 2.3               | 236.59802766605696 | 349.63580114185936 | 2.28997589 |
| 2.4               | 230.73226236763298 | 335.4021735990462  | 2.21146264 |
| 2.5               | 225.81063208450638 | 322.1283767570811  | 2.14038675 |
| 2.6               | 221.61042433293028 | 309.71152104410737 | 2.07547635 |
| 2.7               | 217.98034240765568 | 298.0633635462919  | 2.01579573 |
| 2.8               | 214.8124996751394  | 287.1077569484564  | 1.960626   |
| 2.9               | 212.02620699629517 | 276.7786173540166  | 1.90939385 |
| 3.0               | 209.55912671601234 | 267.0182914050078  | 1.86163054 |
| 3.1               | 207.36199723365263 | 257.7762336924401  | 1.81694621 |
| 3.2               | 205.3952226411796  | 249.0079274825673  | 1.77501231 |
| 3.3               | 203.6265417951376  | 240.67399785651475 | 1.73554898 |
| 3.4               | 202.02937207957373 | 232.7394782121657  | 1.69831582 |
| 3.5               | 200.5815982852197  | 225.17319990633018 | 1.66310468 |
| 3.6               | 199.26466694507747 | 217.9472814559404  | 1.62973417 |
| 3.7               | 198.062896703667   | 211.03669875668663 | 1.59804529 |
| 3.8               | 196.96294518951674 | 204.4189216344791  | 1.56789792 |
| 3.9               | 195.95339152219043 | 198.0736050203686  | 1.53916796 |
| 4.0               | 195.02440568563682 | 191.9823253518871  | 1.51174504 |
| 4.1               | 194.16748409086918 | 186.12835461354484 | 1.48553062 |
| 4.2               | 193.37523620116647 | 180.49646585517428 | 1.46043634 |
| 4.3               | 192.64121098311134 | 175.07276515745124 | 1.43638272 |
| 4.4               | 191.95975472424664 | 169.84454591578037 | 1.41329805 |
| 4.5               | 191.32589377331908 | 164.80016203722548 | 1.39111741 |
| 4.6               | 190.73523724200567 | 159.92891722870021 | 1.36978185 |
| 4.7               | 190.18389581206824 | 155.22096802773802 | 1.34923775 |
| 4.8               | 189.66841362466482 | 150.66723861260056 | 1.32943614 |
| 4.9               | 189.18571086257487 | 146.2593457439868  | 1.31033225 |
| 5.0               | 188.73303512329946 | 141.98953245003423 | 1.29188503 |



### Exact Z for Ising n=16, T from 2.0 to 5.0

| T                 | $lnZ$              | fix                | sum/n      |
| ----------------- | ------------------ | ------------------ | ---------- |
| 2.0               | 1051.104987614517  | 1597.682970064889  | 2.58670699 |
| 2.1               | 1009.4996784717939 | 1525.9571863911228 | 2.47603209 |
| 2.2               | 973.8408186679096  | 1459.8057676137014 | 2.37660799 |
| 2.269185314213022 | 952.6480795485427  | 1416.9390115767155 | 2.31404989 |
| 2.3               | 944.3823241807323  | 1398.5480483845513 | 2.28801794 |
| 2.4               | 920.8489585246118  | 1341.612857520735  | 2.20943537 |
| 2.5               | 901.1617063065412  | 1288.5170967249728 | 2.13835821 |
| 2.6               | 884.3618952526879  | 1238.8491888829687 | 2.07344832 |
| 2.7               | 869.8418508112793  | 1192.2561471669637 | 2.01376758 |
| 2.8               | 857.1705412103629  | 1148.4333700194834 | 1.95859757 |
| 2.9               | 846.0253831110782  | 1107.1165118352076 | 1.90736513 |
| 3.0               | 836.1570646028667  | 1068.0749509785837 | 1.85960158 |
| 3.1               | 827.3685472309867  | 1031.1064990650011 | 1.81491704 |
| 3.2               | 819.5014489849076  | 996.0330835834103  | 1.77298294 |
| 3.3               | 812.4267256294704  | 962.697200232009   | 1.73351946 |
| 3.4               | 806.0380467741903  | 930.958978748977   | 1.69628616 |
| 3.5               | 800.2469515985453  | 900.6937413401133  | 1.6610749  |
| 3.6               | 794.979226238446   | 871.7899593705057  | 1.62770428 |
| 3.7               | 790.1721452729342  | 844.1475341479485  | 1.59601531 |
| 3.8               | 785.7723392163707  | 817.676343055014   | 1.56586785 |
| 3.9               | 781.7341245470767  | 792.2950041887631  | 1.53713782 |
| 4.0               | 778.0181812008658  | 767.929821917147   | 1.50971485 |
| 4.1               | 774.5904948217963  | 744.5138830008635  | 1.48350037 |
| 4.2               | 771.4215032629859  | 721.9862786337237  | 1.45840604 |
| 4.3               | 768.4854023907654  | 700.29143227748    | 1.43435238 |
| 4.4               | 765.7595773553068  | 679.378516774791   | 1.41126767 |
| 4.5               | 763.2241335515964  | 659.2009471181697  | 1.38908699 |
| 4.6               | 760.8615074263428  | 639.7159375871064  | 1.36775141 |
| 4.7               | 758.6561417065932  | 620.884113858062   | 1.34720728 |
| 4.8               | 756.5942129569794  | 602.669172233921   | 1.32740565 |
| 4.9               | 754.6634019086196  | 585.0375794016054  | 1.30830174 |
| 5.0               | 752.8526989515179  | 567.9583071642983  | 1.2898545  |

### Exact Z for Ising n=16, T from 2.0 to 5.0

| T                 | $lnZ$              | fix                | sum/n      |
| ----------------- | ------------------ | ------------------ | ---------- |
| 2.0               | 4202.340507978943  | 6390.731888891347  | 2.58619932 |
| 2.1               | 4035.919151643693  | 6103.828751753977  | 2.47552439 |
| 2.2               | 3893.282375146562  | 5839.223074928786  | 2.37609996 |
| 2.269185314213022 | 3808.672283419809  | 5667.756049896794  | 2.31358114 |
| 2.3               | 3775.475780779397  | 5594.19219679645   | 2.2875166  |
| 2.4               | 3681.3156013825655 | 5366.4514324725105 | 2.2089275  |
| 2.5               | 3602.567357075173  | 5154.068388663982  | 2.13785052 |
| 2.6               | 3535.368138799464  | 4955.39675684231   | 2.07294065 |
| 2.7               | 3477.2879616861483 | 4769.024589647048  | 2.0132599  |
| 2.8               | 3426.6027232992783 | 4593.733480813681  | 1.95808989 |
| 2.9               | 3382.022090902617  | 4428.466047896597  | 1.90685746 |
| 3.0               | 3342.5488168697866 | 4272.299804336221  | 1.8590939  |
| 3.1               | 3307.3947473822673 | 4124.425996581825  | 1.81440936 |
| 3.2               | 3275.9263543979505 | 3984.13233458023   | 1.77247527 |
| 3.3               | 3247.627460976202  | 3850.788801117803  | 1.73301178 |
| 3.4               | 3222.0727455550814 | 3723.8359151425634 | 1.69577848 |
| 3.5               | 3198.908364852501  | 3602.7749654742397 | 1.66056722 |
| 3.6               | 3177.837463412104  | 3487.159837570642  | 1.62719661 |
| 3.7               | 3158.609139550057  | 3376.590136661067  | 1.59550764 |
| 3.8               | 3141.009915323803  | 3270.7053722743944 | 1.56536018 |
| 3.9               | 3124.857056646627  | 3169.180016797821  | 1.53663014 |
| 4.0               | 3109.9932832617833 | 3071.719287702361  | 1.50920717 |
| 4.1               | 3096.282537745505  | 2978.055532030211  | 1.48299269 |
| 4.2               | 3083.6065715102636 | 2887.9451145561466 | 1.45789836 |
| 4.3               | 3071.862168021382  | 2801.1657291268675 | 1.4338447  |
| 4.4               | 3060.958867879547  | 2717.514067112701  | 1.41075999 |
| 4.5               | 3050.817092664706  | 2636.8037884835244 | 1.38857932 |
| 4.6               | 3041.366588163691  | 2558.863750357128  | 1.36724373 |
| 4.7               | 3032.545125284693  | 2483.5364554392577 | 1.3466996  |
| 4.8               | 3024.2974102862377 | 2410.676688941371  | 1.32689797 |
| 4.9               | 3016.5741660927984 | 2340.150317610988  | 1.30779406 |
| 5.0               | 3009.331354264392  | 2271.833228660903  | 1.28934682 |




