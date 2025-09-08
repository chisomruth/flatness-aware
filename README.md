# Flatness-Aware Regularization Experiments

This repository contains the implementation and experiments for Flatness-Aware Regularization (FA-Regularization), a novel regularization technique that penalizes the trace of the squared Hessian (Tr(H²)) to encourage convergence to flatter minima in neural networks.

## Overview

The project implements and evaluates FA-Regularization using:
- 2-layer MLP with ReLU activation on CIFAR-100
- Hutchinson's stochastic trace estimator for efficient Tr(H²) computation
- Comprehensive experiments across different regularization strengths
- Loss landscape visualization to demonstrate flatness


## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the complete experiment suite:
```bash
python main.py
```

This will:
1. Load and preprocess CIFAR-100 data
2. Train models with different λ values [0.0, 0.001, 0.01, 0.1, 1.0]
3. Generate training comparison plots
4. Visualize loss landsFApes
5. Print comprehensive analysis
6. Save results to `FA_regularization_results.pkl`
7. Generate markdown report `experiment_report.md`

### Configuration

Modify experiment parameters in `config.py`:

```python
EXPERIMENT_CONFIG = {
    'n_samples': 3000,           # Dataset size
    'learning_rate': 0.001,      # Learning rate
    'num_epochs': 50,            # Training epochs
    'batch_size': 64,            # Batch size
    'lambda_values': [0.0, 0.001, 0.01, 0.1, 1.0],  # Regularization strengths
    'num_hutchinson_samples': 3,  # Hutchinson estimator samples
}
```

### Individual Components

You can also run individual components:

```python
from data_utils import load_cifar100_subset
from model import TwoLayerMLP
from training import train_model

# Load data
X_train, y_train, X_test, y_test = load_cifar100_subset(n_samples=1000)

# Create model
model = TwoLayerMLP(input_dim=3072, hidden_dim=256, output_dim=100, key=jax.random.PRNGKey(42))

# Train with FA-regularization
result = train_model(X_train, y_train, X_test, y_test, model, lambda_reg=0.01)
```

## Algorithm Implementation

The FA-Regularization loss function is implemented as:

```
L_total = L_task + λ · (Tr(H²) / B)
```

Where:
- `L_task`: Task-specific loss (cross-entropy)
- `λ`: Regularization strength
- `Tr(H²)`: Trace of squared Hessian (estimated via Hutchinson)
- `B`: Batch size (for normalization)

## Experiments

The code runs comprehensive experiments to evaluate:

### 1. Generalization Performance
- Test accuracy comparison across different λ values
- Training loss convergence analysis

### 2. Training Stability
- Loss variance in final epochs
- Accuracy variance analysis

### 3. Convergence Speed
- Epochs to reach target accuracy
- Training time comparison

### 4. Loss Landscape Analysis
- 2D loss landscape visualization
- Quantitative sharpness metrics
- Flatness comparison between baseline and FA-regularized models

## Expected Outputs

### Visualizations
1. 4-panel training comparison: Test accuracy, training loss, Flatness estimates, final metrics
2. 3-panel loss landsFApes: Baseline vs FA-regularized vs difference

### Analysis
1. Performance summary table: Final accuracy, best accuracy, average Flatness, training time
2. Stability metrics: Variance analysis for training stability
3. Convergence analysis: Speed comparison across λ values
4. Research report: Comprehensive markdown report with findings

### Files Generated
- `FA_regularization_results.pkl`: Complete experimental results
- `experiment_report.md`: Detailed analysis report

## Research Questions Addressed

1. Does FA-regularization improve generalization?
   - Compare test accuracies across λ values

2. Does it find flatter minima?
   - Loss landscape visualization and sharpness metrics

3. What's the optimal regularization strength?
   - Performance analysis across different λ values

4. Is training stable with Flatness penalties?
   - Variance analysis and convergence behavior

## Technical Details

### Hutchinson's Estimator
- Uses Gaussian random vectors for unbiased Tr(H²) estimation
- Configurable number of samples for accuracy/efficiency trade-off
- Automatic differentiation for Hessian-vector products

### Loss LandsFApe Visualization
- Random orthogonal directions for 2D projections
- Batch normalization for fair comparison
- Quantitative sharpness metrics

### Reproducibility
- Fixed random seeds for consistent results
- Same model initialization across experiments
- Comprehensive logging of all parameters

## Hardware Requirements

- CPU: Any modern CPU (experiments optimized for CPU execution)
- RAM: 4GB+ recommended
- Storage: ~100MB for results and visualizations

For GPU acceleration, modify `JAX_CONFIG['platform'] = 'gpu'` in `config.py`.

## Customization

### Different Datasets
Modify `data_utils.py` to load different datasets:
```python
def load_custom_dataset():
    # Your dataset loading logic
    return X_train, y_train, X_test, y_test
```

### Different Models
Extend `model.py` for different architectures:
```python
class ConvNet:
    # Your model implementation
```

### Additional Metrics
Add custom analysis in `analysis.py`:
```python
def custom_analysis(results):
    # Your analysis logic
```

## Citation

If you use this code in your research, please cite:

```
@article{chibfatim2025Flatness,
    title={Flatness-Aware Regularization for Robust Generalization in Deep Networks},
    author={Chibuike Chisom and Adebanjo Fatimo},
    journal=unpublished,
    year={2025}
}
```

## License

MIT License - see LICENSE file for details.
