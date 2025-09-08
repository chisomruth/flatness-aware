import jax
import numpy as np
from jax import random
import time
import pickle
from config import EXPERIMENT_CONFIG, JAX_CONFIG
from utils.data_loader import load_cifar100_subset
from utils.model import TwoLayerMLP
from utils.training_loop import train_model
from utils.visuals import plot_training_comparison, plot_loss_landscapes
from utils.reports import print_experiment_summary, analyze_training_stability, generate_research_report

def setup_environment():
    np.random.seed(EXPERIMENT_CONFIG['numpy_seed'])
    key = random.PRNGKey(EXPERIMENT_CONFIG['jax_seed'])

    jax.config.update('jax_platform_name', JAX_CONFIG['platform'])
    
    print("JAX backend:", jax.default_backend())
    print("JAX devices:", jax.devices())
    
    return key

def run_single_experiment(X_train, y_train, X_test, y_test, lambda_reg, config):
    print(f"\n--- Training with λ = {lambda_reg} ---")
    
    key_model = random.PRNGKey(config['jax_seed'])
    model = TwoLayerMLP(X_train.shape[1], config['hidden_dim'], config['output_dim'], key_model)

    start_time = time.time()
    result = train_model(
        X_train, y_train, X_test, y_test, model,
        lambda_reg=lambda_reg,
        num_hutchinson_samples=config['num_hutchinson_samples'],
        learning_rate=config['learning_rate'],
        num_epochs=config['num_epochs'],
        batch_size=config['batch_size']
    )
    total_time = time.time() - start_time
    
    result['total_training_time'] = total_time
    result['lambda'] = lambda_reg
    
    print(f"Final test accuracy: {result['final_test_accuracy']:.4f}")
    print(f"Total training time: {total_time:.2f}s")
    
    return result

def run_experiments():
    
    key = setup_environment()
    config = EXPERIMENT_CONFIG
    
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test = load_cifar100_subset(n_samples=config['n_samples'])
    
    results = {}
    trained_models = {}
    
    print("="*60)
    print("FLATNESS-AWARE REGULARIZATION EXPERIMENTS")
    print("="*60)
    
    for lambda_reg in config['lambda_values']:
        result = run_single_experiment(X_train, y_train, X_test, y_test, lambda_reg, config)
        
        results[lambda_reg] = result
        
        key_model = random.PRNGKey(config['jax_seed'])
        model = TwoLayerMLP(X_train.shape[1], config['hidden_dim'], config['output_dim'], key_model)
        trained_models[lambda_reg] = (model, result['params'])
    
    return results, trained_models, X_test, y_test

def main():
    
    print("Starting Flatness-Aware Regularization Experiments...")
    print("=" * 80)
    
    results, trained_models, X_test, y_test = run_experiments()
    
    print("\n--- Generating Training Comparison Plots ---")
    plot_training_comparison(results)
    
    print("\n--- Computing Loss Landscapes ---")
    plot_loss_landscapes(trained_models, X_test, y_test)
    
    print("\n--- Generating Analysis ---")
    print_experiment_summary(results)
    analyze_training_stability(results)
    
    report = generate_research_report(results, {
        'dataset': 'CIFAR-100 subset',
        'model': '2-layer MLP with ReLU',
        'hidden_dim': EXPERIMENT_CONFIG['hidden_dim'],
        'num_epochs': EXPERIMENT_CONFIG['num_epochs'],
        'batch_size': EXPERIMENT_CONFIG['batch_size'],
        'learning_rate': EXPERIMENT_CONFIG['learning_rate'],
        'num_hutchinson_samples': EXPERIMENT_CONFIG['num_hutchinson_samples']
    })
   
    if EXPERIMENT_CONFIG['save_results']:
        print(f"\n--- Saving Results ---")
        
        # Save pickle file (this should work fine)
        try:
            with open(EXPERIMENT_CONFIG['results_filename'], 'wb') as f:
                pickle.dump({
                    'results': results,
                    'experiment_config': EXPERIMENT_CONFIG,
                    'trained_models': {k: v[1] for k, v in trained_models.items()},  # Save only parameters
                }, f)
            print(f"Results saved to '{EXPERIMENT_CONFIG['results_filename']}'")
        except Exception as e:
            print(f"Error saving pickle file: {e}")
        
        try:
            with open(EXPERIMENT_CONFIG['report_filename'], 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Research report saved to '{EXPERIMENT_CONFIG['report_filename']}'")
        except UnicodeEncodeError as e:
            print(f"Unicode encoding error: {e}")
            print("Attempting to save with ASCII-compatible characters...")
            # Fallback: replace Greek letters with ASCII equivalents
            report_ascii = report.replace('λ', 'lambda').replace('α', 'alpha').replace('β', 'beta').replace('γ', 'gamma')
            fallback_filename = EXPERIMENT_CONFIG['report_filename'].replace('.txt', '_ascii.txt')
            with open(fallback_filename, 'w', encoding='utf-8') as f:
                f.write(report_ascii)
            print(f"Report saved with ASCII characters to '{fallback_filename}'")
        except Exception as e:
            print(f"Error saving report: {e}")
    
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return results, trained_models

if __name__ == "__main__":
    results, trained_models = main()