# Code for Adaptive and Extragradient Algorithms on GAN

This is the code associated with the paper [A Universal Algorithm for Variational Inequalities Adaptive to Smoothness and Noise](https://arxiv.org/abs/1902.01637) and the proposed algortihm Universal Mirror Prox(UMP).

## Requirements

The code is in `pytorch` and was tested for:
- pytorch=0.4.0

## `class UMP`

The UMP method is packaged as a `torch.optim.Optimizer` with an additional method `extrapolation()`. 

Example of how to run `UMP`:
```python
for i, input, target in enumerate(dataset):
    UMP.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    if i%2:
      UMP.extrapolation()
    else:
      UMP.step()
```

## Experiments

To run the WGAN-GP experiment with Universal

To run the WGAN-GP experiment with ExtraAdam and the ResNet architecture on CIFAR10 with the parameters from the paper:
`python train_extraadam.py results\ --default --model resnet --cuda`

The `--default` option loads the hyperparameters used in the paper for each experiments, they are available as JSON files in the `config` folder.

The weights for our WGAN-GP and ResNet model trained with ExtraAdam is available in the `results` folder.

For evaluation:
`python eval_inception_score.py results/ExtraAdam/best_model.state`

A `ipython` notebook is also available for the bilinear example [here](bilinear_wgan.ipynb).

## Results
with Averaging:

![AvgExtraAdam samples on CIFAR10 for ResNet WGAN-GP](results/ExtraAdam/gen_averaging/500000.png)

without Averaging:

![ExtraAdam samples on CIFAR10 for ResNet WGAN-GP](results/ExtraAdam/gen/500000.png)
