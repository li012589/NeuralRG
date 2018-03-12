

# NerualRG 

Pytorch implement of arXiv paper: [Neural Network Renormalization Group](https://arxiv.org/abs/1802.02840).

**NeuralRG** is a deep generative model using variational renormalization group approach, it is composed by layers of bijectors(In our demo, we use [realNVP](https://arxiv.org/abs/1605.08803)). After training, it can directly generate statistically independent physical configurations.

## How NerualRG Network

### Hierarchy Bijectors 

In NerualRG Network(a), we use realNVP(b) networks as building blocks, realNVP is a kind of bijectors, they can transform one distribution into other distribution and revert this process. For multi-in-multi-out blocks, we call they disentanglers(gray blocks in (a)), and for multi-in-single-out blocks, we can they decimators(white blocks in (a)). And stacking  multiply layers of these blocks into a hierarchical structure forms NerualRG network, so NerualRG is also a bijector. In inference process, each layer try to "separate" entangled variables into independent variables, and at layers composed of decimators, we only keep one of these independent variables, this is renormalization group.

![NerualRG Network](etc/Nflow.png)

The result of renormalization group is that ,in generating process, at shallow layers of NerualRG Network, the configuration formed by output variables is a "blurry" version of deep layers' output. This can be seen from following training results.

![2D Ising Configuration](etc/rg.png)

![MNIST](etc/mnist.png)

### Training



## How to Run 

### Requirements

* pytorch
* numpy
* matplotlib

### TEBD-like Structure

### MERA-like Structure

## Results in the paper

See [notebook](etc/paper.md).

## Citation

If you use this code for your research, please cite our [paper](https://arxiv.org/abs/1802.02840):

```
@article{nerualRG,
  Author = {Shuo-Hui Li and Lei Wang},
  Title = {Neural Network Renormalization Group},
  Year = {2018},
  Eprint = {arXiv:1802.02840},
}
```

## Contact

Feel free to contact me at: [contact_lish@iphy.ac.cn](mailto:contact_lish@iphy.ac.cn).






