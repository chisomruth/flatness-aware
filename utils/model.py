import jax
import jax.numpy as jnp
from jax import random
from typing import Dict

class TwoLayerMLP:
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, key: jax.random.PRNGKey):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        k1, k2 = random.split(key)
        self.params = {
            'W1': random.normal(k1, (input_dim, hidden_dim)) * jnp.sqrt(2.0 / input_dim),
            'b1': jnp.zeros(hidden_dim),
            'W2': random.normal(k2, (hidden_dim, output_dim)) * jnp.sqrt(2.0 / hidden_dim),
            'b2': jnp.zeros(output_dim)
        }
    
    def forward(self, params: Dict, x: jnp.ndarray):
        h1 = jax.nn.relu(x @ params['W1'] + params['b1'])
        logits = h1 @ params['W2'] + params['b2']
        return logits
    
    def predict(self, params: Dict, x: jnp.ndarray):
        logits = self.forward(params, x)
        return jnp.argmax(logits, axis=-1)