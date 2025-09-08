
EXPERIMENT_CONFIG = {
    'n_samples': 3000,
    'train_test_split': 0.8,
    
    'hidden_dim': 256,
    'output_dim': 100,
    
    'learning_rate': 0.001,
    'num_epochs': 50,
    'batch_size': 64,
    'num_hutchinson_samples': 3,
    
    'lambda_values': [0.0, 0.001, 0.01, 0.1, 1.0],
    

    'landscape_range': 1.0,
    'landscape_resolution': 50,
    'subset_size_for_landscape': 500,
    
    'numpy_seed': 42,
    'jax_seed': 42,
    
    'verbose': True,
    'save_results': True,
    'results_filename': 'ca_regularization_results.pkl',
    'report_filename': 'experiment_report.md'
}

JAX_CONFIG = {
    'platform': 'cpu',  # Change to 'gpu' if available
    'precision': 'float32'
}