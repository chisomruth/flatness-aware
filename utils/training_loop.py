import time
import jax.numpy as jnp
from jax import grad, random
import optax
from typing import Dict
from utils.model import TwoLayerMLP
from utils.loss_func import ca_loss_fn, cross_entropy_loss, hutchinson_trace_estimator

def train_model(X_train: jnp.ndarray, y_train: jnp.ndarray, X_test: jnp.ndarray, y_test: jnp.ndarray,
                model: TwoLayerMLP, lambda_reg: float = 0.0, num_hutchinson_samples: int = 5,
                learning_rate: float = 0.01, num_epochs: int = 100, batch_size: int = 128):
    params = model.params.copy()
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    train_losses = []
    test_accuracies = []
    curvature_estimates = []
    training_times = []

    n_batches = len(X_train) // batch_size
    
    print(f"Training with λ={lambda_reg}, {num_epochs} epochs, {n_batches} batches per epoch")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_curvature = 0.0
        
        key_epoch = random.PRNGKey(epoch)
        perm = random.permutation(key_epoch, len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            key_batch = random.PRNGKey(epoch * n_batches + batch_idx)
            
            def loss_fn(p):
                return ca_loss_fn(p, model.forward, X_batch, y_batch, 
                                  lambda_reg, num_hutchinson_samples, key_batch)

            loss_value = loss_fn(params)
            grads = grad(loss_fn)(params)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
            epoch_loss += loss_value
            
            if lambda_reg > 0:
                def task_loss_only(p, xb, yb):
                    return cross_entropy_loss(p, model.forward, xb, yb)
                
                curvature_est = hutchinson_trace_estimator(
                    params, task_loss_only, X_batch, y_batch, 
                    num_hutchinson_samples, key_batch
                )
                epoch_curvature += curvature_est / batch_size
        

        test_logits = model.forward(params, X_test)
        test_preds = jnp.argmax(test_logits, axis=-1)
        test_acc = jnp.mean(test_preds == y_test)
        
        avg_loss = epoch_loss / n_batches
        avg_curvature = epoch_curvature / n_batches if lambda_reg > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        
        train_losses.append(float(avg_loss))
        test_accuracies.append(float(test_acc))
        curvature_estimates.append(float(avg_curvature))
        training_times.append(epoch_time)
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Test Acc={test_acc:.4f}, "
                  f"Flatness={avg_curvature:.6f}, Time={epoch_time:.2f}s")
    
    return {
        'params': params,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'flatness_estimates': curvature_estimates,
        'training_times': training_times,
        'final_test_accuracy': float(test_accuracies[-1])
    }
