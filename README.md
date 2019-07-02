# Code for Adaptive and Extragradient Algorithms on GAN

This is the code associated with the paper [A Universal Algorithm for Variational Inequalities Adaptive to Smoothness and Noise](https://arxiv.org/abs/1902.01637) by Francis Bach and Kfir Y. Levy, and the proposed algortihm Universal Mirror Prox (UMP) in Euclidean space.

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

Note the averaging is taken at the `extrapolation` step.

## Experiments

To run the WGAN-GP experiment with UMP and the DCGAN architecture on CIFAR10:
`python train.py -alg UMP --model dcgan --cuda`

The `--default` option loads the hyperparameters used in the paper for each experiments, they are available as JSON files in the `config` folder.

The weights for our WGAN-GP and ResNet model trained with ExtraAdam is available in the `results` folder.

## Results
with Averaging:

![AvgExtraAdam samples on CIFAR10 for ResNet WGAN-GP](results/ExtraAdam/gen_averaging/500000.png)

without Averaging:

![ExtraAdam samples on CIFAR10 for ResNet WGAN-GP](results/ExtraAdam/gen/500000.png)
