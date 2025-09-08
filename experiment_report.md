
    # Flatness-Aware Regularization Experiment Report

    ## Experimental Setup
    - Dataset: CIFAR-100 subset
- Model: 2-layer MLP with ReLU
- Hidden dimensions: 256
- Training epochs: 50
- Batch size: 64
- Learning rate: 0.001
- Hutchinson samples: 3

## Results Summary

- **Best λ value**: 0.01
- **Best test accuracy**: 0.2700
- **Baseline accuracy**: 0.2633
- **Performance improvement**: 2.53%

## Detailed Results

| Lambda | Final Acc | Best Acc | Avg Flatness | Train Time |
|--------|-----------|----------|---------------|------------|
| 0.000 | 0.2633 | 0.2800 | 0.000000 | 60.85s |
| 0.001 | 0.2633 | 0.2833 | 23.050173 | 2290.77s |
| 0.010 | 0.2700 | 0.2783 | 10.739012 | 2433.53s |
| 0.100 | 0.2667 | 0.2967 | 4.144254 | 2203.87s |
| 1.000 | 0.2583 | 0.2983 | 1.549933 | 2337.86s |

## Analysis

### Training Stability

- λ=0.0: Loss variance=0.000050, Accuracy variance=0.000092
- λ=0.001: Loss variance=0.000052, Accuracy variance=0.000064
- λ=0.01: Loss variance=0.000097, Accuracy variance=0.000042
- λ=0.1: Loss variance=0.000478, Accuracy variance=0.000019
- λ=1.0: Loss variance=0.009053, Accuracy variance=0.000087

### Convergence Speed

Time to reach 0.2502 accuracy:
- λ=0.0: 0 epochs
- λ=0.001: 0 epochs
- λ=0.01: 0 epochs
- λ=0.1: 0 epochs
- λ=1.0: 13 epochs
