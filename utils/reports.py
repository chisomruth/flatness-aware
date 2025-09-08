import numpy as np
from typing import Dict

def print_experiment_summary(results: Dict):

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"{'Lambda':<10} {'Final Acc':<12} {'Best Acc':<12} {'Avg flatness':<15} {'Train Time':<12}")
    print("-" * 70)
    
    for lambda_reg, result in results.items():
        final_acc = result['final_test_accuracy']
        best_acc = max(result['test_accuracies'])
        avg_curvature = np.mean(result['flatness_estimates']) if lambda_reg > 0 else 0.0
        train_time = result['total_training_time']
        
        print(f"{lambda_reg:<10.3f} {final_acc:<12.4f} {best_acc:<12.4f} {avg_curvature:<15.6f} {train_time:<12.2f}")
    

    best_lambda = max(results.keys(), key=lambda l: results[l]['final_test_accuracy'])
    best_acc = results[best_lambda]['final_test_accuracy']
    baseline_acc = results[0.0]['final_test_accuracy']
    
    print(f"\nBest performing λ: {best_lambda}")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print(f"Improvement: {((best_acc - baseline_acc) / baseline_acc * 100):.2f}%")
    
    print(f"\nConvergence Analysis:")
    target_acc = 0.95 * baseline_acc 
    
    for lambda_reg, result in results.items():
        accuracies = result['test_accuracies']
        epochs_to_target = None
        for epoch, acc in enumerate(accuracies):
            if acc >= target_acc:
                epochs_to_target = epoch
                break
        
        if epochs_to_target is not None:
            print(f"λ={lambda_reg}: Reached {target_acc:.4f} accuracy at epoch {epochs_to_target}")
        else:
            print(f"λ={lambda_reg}: Did not reach target accuracy")

def analyze_training_stability(results: Dict):
    print("\n--- Training Stability Analysis ---")
    
    for lambda_reg, result in results.items():
        losses = np.array(result['train_losses'])
        accuracies = np.array(result['test_accuracies'])

        loss_variance = np.var(losses[-10:])  
        acc_variance = np.var(accuracies[-10:]) 
        
        print(f"λ={lambda_reg}: Loss variance (last 10 epochs): {loss_variance:.6f}")
        print(f"λ={lambda_reg}: Accuracy variance (last 10 epochs): {acc_variance:.6f}")

def generate_research_report(results: Dict, config: Dict) -> str:
    report = """
    # Flatness-Aware Regularization Experiment Report

    ## Experimental Setup
    """
    
    report += f"- Dataset: {config['dataset']}\n"
    report += f"- Model: {config['model']}\n"
    report += f"- Hidden dimensions: {config['hidden_dim']}\n"
    report += f"- Training epochs: {config['num_epochs']}\n"
    report += f"- Batch size: {config['batch_size']}\n"
    report += f"- Learning rate: {config['learning_rate']}\n"
    report += f"- Hutchinson samples: {config['num_hutchinson_samples']}\n"
    
    report += "\n## Results Summary\n\n"
    
    best_lambda = max(results.keys(), key=lambda l: results[l]['final_test_accuracy'])
    best_acc = results[best_lambda]['final_test_accuracy']
    baseline_acc = results[0.0]['final_test_accuracy']
    improvement = ((best_acc - baseline_acc) / baseline_acc * 100)
    
    report += f"- **Best λ value**: {best_lambda}\n"
    report += f"- **Best test accuracy**: {best_acc:.4f}\n"
    report += f"- **Baseline accuracy**: {baseline_acc:.4f}\n"
    report += f"- **Performance improvement**: {improvement:.2f}%\n"
    
    report += "\n## Detailed Results\n\n"
    

    report += "| Lambda | Final Acc | Best Acc | Avg Flatness | Train Time |\n"
    report += "|--------|-----------|----------|---------------|------------|\n"
    
    for lambda_reg, result in results.items():
        final_acc = result['final_test_accuracy']
        best_acc = max(result['test_accuracies'])
        avg_curvature = np.mean(result['flatness_estimates']) if lambda_reg > 0 else 0.0
        train_time = result['total_training_time']
        
        report += f"| {lambda_reg:.3f} | {final_acc:.4f} | {best_acc:.4f} | {avg_curvature:.6f} | {train_time:.2f}s |\n"
    
    report += "\n## Analysis\n\n"
    
    report += "### Training Stability\n\n"
    for lambda_reg, result in results.items():
        losses = np.array(result['train_losses'])
        accuracies = np.array(result['test_accuracies'])
        loss_var = np.var(losses[-10:])
        acc_var = np.var(accuracies[-10:])
        
        report += f"- λ={lambda_reg}: Loss variance={loss_var:.6f}, Accuracy variance={acc_var:.6f}\n"
    
    report += "\n### Convergence Speed\n\n"
    target_acc = 0.95 * baseline_acc
    report += f"Time to reach {target_acc:.4f} accuracy:\n"
    
    for lambda_reg, result in results.items():
        accuracies = result['test_accuracies']
        epochs_to_target = None
        for epoch, acc in enumerate(accuracies):
            if acc >= target_acc:
                epochs_to_target = epoch
                break
        
        if epochs_to_target is not None:
            report += f"- λ={lambda_reg}: {epochs_to_target} epochs\n"
        else:
            report += f"- λ={lambda_reg}: Did not reach target\n"
    
    return report