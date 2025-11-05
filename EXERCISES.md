# Practical Exercises - Optimization Algorithms
## Hands-On Problems for Interview Preparation

---

## 🎯 Purpose

These exercises will help you:
1. Solidify understanding through practice
2. Prepare for coding interviews
3. Build confidence in implementation
4. Develop debugging skills

**Difficulty levels:** ⭐ Easy | ⭐⭐ Medium | ⭐⭐⭐ Hard

---

## Part 1: Gradient Computation

### Exercise 1.1: Simple Derivatives ⭐
**Task:** Compute gradients analytically for these functions:

a) `f(x) = x²`
   - Compute: ∂f/∂x = ?
   - Verify: At x=3, gradient = ?

b) `f(x, y) = x² + 3xy + y²`
   - Compute: ∂f/∂x = ?
   - Compute: ∂f/∂y = ?

c) `f(θ) = ||θ - [1, 2, 3]||²` where θ ∈ ℝ³
   - Compute: ∇f(θ) = ?

**Solution hints in `MATHEMATICAL_DERIVATIONS.md`**

---

### Exercise 1.2: Numerical Gradient ⭐⭐
**Task:** Implement numerical gradient computation

```python
def numerical_gradient(f, x, epsilon=1e-5):
    """
    Compute gradient of f at x numerically.
    
    Args:
        f: Function to differentiate
        x: Point at which to compute gradient (numpy array)
        epsilon: Small step size
    
    Returns:
        grad: Numerical gradient (same shape as x)
    """
    # YOUR CODE HERE
    pass

# Test cases
def f1(x):
    return np.sum(x**2)

def f2(x):
    return np.sum((x - np.array([1, 2, 3]))**2)

x_test = np.array([2.0, 3.0, 4.0])
grad1 = numerical_gradient(f1, x_test)
grad2 = numerical_gradient(f2, x_test)

print(f"Gradient of f1 at {x_test}: {grad1}")
print(f"Expected: {2*x_test}")  # Should be [4, 6, 8]

print(f"\nGradient of f2 at {x_test}: {grad2}")
print(f"Expected: {2*(x_test - np.array([1,2,3]))}")  # Should be [2, 2, 2]
```

**Expected output:**
```
Gradient of f1 at [2. 3. 4.]: [4. 6. 8.]
Expected: [4. 6. 8.]

Gradient of f2 at [2. 3. 4.]: [2. 2. 2.]
Expected: [2. 2. 2.]
```

---

### Exercise 1.3: Gradient Descent from Scratch ⭐⭐
**Task:** Implement basic gradient descent

```python
def gradient_descent(f, grad_f, x0, learning_rate=0.1, n_iterations=100):
    """
    Minimize function f using gradient descent.
    
    Args:
        f: Function to minimize
        grad_f: Function that computes gradient of f
        x0: Initial point
        learning_rate: Step size
        n_iterations: Number of iterations
    
    Returns:
        x: Final point
        history: List of (x, f(x)) at each iteration
    """
    # YOUR CODE HERE
    pass

# Test: Minimize f(x, y) = (x-1)² + (y-2)²
def f(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def grad_f(x):
    return np.array([2*(x[0] - 1), 2*(x[1] - 2)])

x0 = np.array([5.0, 5.0])
x_final, history = gradient_descent(f, grad_f, x0, learning_rate=0.1, n_iterations=50)

print(f"Starting point: {x0}")
print(f"Optimal point: [1, 2]")
print(f"Found point: {x_final}")
print(f"Distance from optimal: {np.linalg.norm(x_final - np.array([1, 2])):.6f}")
```

---

## Part 2: Neural Network Components

### Exercise 2.1: Activation Functions ⭐
**Task:** Implement common activation functions and their derivatives

```python
def tanh(z):
    """Hyperbolic tangent activation."""
    # YOUR CODE HERE
    pass

def tanh_derivative(z):
    """Derivative of tanh."""
    # YOUR CODE HERE
    # Hint: tanh'(z) = 1 - tanh²(z)
    pass

def relu(z):
    """ReLU activation."""
    # YOUR CODE HERE
    pass

def relu_derivative(z):
    """Derivative of ReLU."""
    # YOUR CODE HERE
    pass

def sigmoid(z):
    """Sigmoid activation."""
    # YOUR CODE HERE
    pass

def sigmoid_derivative(z):
    """Derivative of sigmoid."""
    # YOUR CODE HERE
    # Hint: σ'(z) = σ(z)(1 - σ(z))
    pass

# Test cases
z_test = np.array([-2, -1, 0, 1, 2])
print(f"tanh({z_test}) = {tanh(z_test)}")
print(f"ReLU({z_test}) = {relu(z_test)}")
print(f"sigmoid({z_test}) = {sigmoid(z_test)}")
```

**Expected behavior:**
- tanh maps to [-1, 1]
- ReLU zeros out negatives
- sigmoid maps to [0, 1]

---

### Exercise 2.2: Forward Propagation ⭐⭐
**Task:** Implement forward pass for a 2-layer network

```python
def forward_pass(X, W1, b1, W2, b2):
    """
    Compute forward pass for 2-layer network.
    
    Architecture:
        h1 = tanh(W1·X + b1)
        y = W2·h1 + b2
    
    Args:
        X: Input (batch_size, input_dim)
        W1: First layer weights (hidden_dim, input_dim)
        b1: First layer bias (hidden_dim,)
        W2: Second layer weights (output_dim, hidden_dim)
        b2: Second layer bias (output_dim,)
    
    Returns:
        y: Output (batch_size, output_dim)
        cache: Dictionary with intermediate values for backprop
    """
    # YOUR CODE HERE
    # Remember to save z1, h1 for backward pass!
    pass

# Test with random data
np.random.seed(42)
X = np.random.randn(5, 10)  # 5 samples, 10 features
W1 = np.random.randn(20, 10)  # 10 → 20
b1 = np.random.randn(20)
W2 = np.random.randn(2, 20)  # 20 → 2
b2 = np.random.randn(2)

y, cache = forward_pass(X, W1, b1, W2, b2)
print(f"Input shape: {X.shape}")
print(f"Output shape: {y.shape}")
print(f"Cache keys: {cache.keys()}")
```

---

### Exercise 2.3: Backward Propagation ⭐⭐⭐
**Task:** Implement backward pass (backpropagation)

```python
def backward_pass(dy, cache, W2):
    """
    Compute gradients using backpropagation.
    
    Args:
        dy: Gradient of loss w.r.t output (batch_size, output_dim)
        cache: Dictionary from forward pass
        W2: Second layer weights
    
    Returns:
        grads: Dictionary with gradients
            - dW2: Gradient w.r.t W2
            - db2: Gradient w.r.t b2
            - dW1: Gradient w.r.t W1
            - db1: Gradient w.r.t b1
    """
    # Extract cached values
    X = cache['X']
    z1 = cache['z1']
    h1 = cache['h1']
    
    # YOUR CODE HERE
    # Hints:
    # 1. δ2 = dy
    # 2. dW2 = δ2·h1ᵀ
    # 3. db2 = sum(δ2)
    # 4. δ1 = (W2ᵀ·δ2) ⊙ tanh'(z1)
    # 5. dW1 = δ1·Xᵀ
    # 6. db1 = sum(δ1)
    
    grads = {}
    # YOUR CODE HERE
    
    return grads

# Test (continuing from Exercise 2.2)
dy = np.random.randn(5, 2)  # Random gradient from loss
grads = backward_pass(dy, cache, W2)

print("Gradient shapes:")
for key, value in grads.items():
    print(f"  {key}: {value.shape}")
```

---

## Part 3: Complete Training Loop

### Exercise 3.1: MSE Loss ⭐
**Task:** Implement mean squared error loss and its gradient

```python
def mse_loss(y_pred, y_true):
    """
    Compute mean squared error.
    
    Args:
        y_pred: Predictions (batch_size, output_dim)
        y_true: True values (batch_size, output_dim)
    
    Returns:
        loss: Scalar MSE value
    """
    # YOUR CODE HERE
    pass

def mse_gradient(y_pred, y_true):
    """
    Compute gradient of MSE w.r.t predictions.
    
    Args:
        y_pred: Predictions (batch_size, output_dim)
        y_true: True values (batch_size, output_dim)
    
    Returns:
        grad: Gradient (batch_size, output_dim)
    """
    # YOUR CODE HERE
    # Hint: ∂MSE/∂y_pred = (2/n)(y_pred - y_true)
    pass

# Test
y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
y_true = np.array([[1.5, 2.5], [2.5, 3.5]])

loss = mse_loss(y_pred, y_true)
grad = mse_gradient(y_pred, y_true)

print(f"MSE Loss: {loss:.4f}")
print(f"Gradient shape: {grad.shape}")
print(f"Gradient:\n{grad}")
```

---

### Exercise 3.2: Mini-Batch Training ⭐⭐⭐
**Task:** Implement complete training loop with mini-batches

```python
def train_network(X_train, Y_train, X_val, Y_val, 
                  hidden_dim=20, learning_rate=0.01, 
                  batch_size=32, n_epochs=100):
    """
    Train a 2-layer neural network.
    
    Args:
        X_train, Y_train: Training data
        X_val, Y_val: Validation data
        hidden_dim: Size of hidden layer
        learning_rate: Step size
        batch_size: Mini-batch size
        n_epochs: Number of epochs
    
    Returns:
        params: Trained parameters
        history: Training history
    """
    # Initialize parameters
    input_dim = X_train.shape[1]
    output_dim = Y_train.shape[1]
    
    W1 = np.random.randn(hidden_dim, input_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(output_dim, hidden_dim) * 0.01
    b2 = np.zeros(output_dim)
    
    n_train = X_train.shape[0]
    history = {'train_loss': [], 'val_loss': []}
    
    # YOUR CODE HERE
    # Implement training loop:
    # 1. For each epoch:
    #    2. Shuffle training data
    #    3. For each mini-batch:
    #       4. Forward pass
    #       5. Compute loss
    #       6. Backward pass
    #       7. Update parameters
    #    8. Evaluate on validation set
    #    9. Store losses
    
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return params, history

# Test with synthetic regression data
np.random.seed(42)
X_train = np.random.randn(500, 10)
Y_train = X_train @ np.random.randn(10, 2) + np.random.randn(500, 2) * 0.1

X_val = np.random.randn(100, 10)
Y_val = X_val @ np.random.randn(10, 2) + np.random.randn(100, 2) * 0.1

params, history = train_network(X_train, Y_train, X_val, Y_val, n_epochs=50)

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training Curves')
plt.show()
```

---

## Part 4: Debugging and Analysis

### Exercise 4.1: Gradient Checking ⭐⭐
**Task:** Verify your backpropagation implementation

```python
def gradient_check(X, Y, params, epsilon=1e-5):
    """
    Verify backpropagation gradients using numerical gradients.
    
    Args:
        X: Input data
        Y: Target data
        params: Network parameters
        epsilon: Small perturbation
    
    Returns:
        max_diff: Maximum relative error
        passed: Boolean indicating if check passed
    """
    # YOUR CODE HERE
    # 1. Compute analytical gradients using backprop
    # 2. Compute numerical gradients for each parameter
    # 3. Compare and compute relative error
    # 4. Return max error and whether test passed
    pass

# Test
np.random.seed(42)
X = np.random.randn(10, 5)
Y = np.random.randn(10, 2)
params = {
    'W1': np.random.randn(8, 5) * 0.01,
    'b1': np.zeros(8),
    'W2': np.random.randn(2, 8) * 0.01,
    'b2': np.zeros(2)
}

max_diff, passed = gradient_check(X, Y, params)
print(f"Maximum relative error: {max_diff:.2e}")
print(f"Gradient check: {'✓ PASSED' if passed else '✗ FAILED'}")
```

**Success criteria:** Relative error < 1e-7

---

### Exercise 4.2: Learning Rate Experimentation ⭐⭐
**Task:** Analyze effect of different learning rates

```python
def compare_learning_rates(X_train, Y_train, learning_rates):
    """
    Train networks with different learning rates and compare.
    
    Args:
        X_train, Y_train: Training data
        learning_rates: List of learning rates to try
    
    Returns:
        results: Dictionary with results for each learning rate
    """
    results = {}
    
    for lr in learning_rates:
        # YOUR CODE HERE
        # Train network with this learning rate
        # Store final loss and convergence behavior
        pass
    
    return results

# Test
learning_rates = [0.001, 0.01, 0.1, 1.0]
results = compare_learning_rates(X_train, Y_train, learning_rates)

# Visualize
plt.figure(figsize=(12, 4))
for lr, history in results.items():
    plt.plot(history, label=f'α={lr}')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.title('Learning Rate Comparison')
plt.show()
```

**Questions to answer:**
1. Which learning rate converges fastest?
2. Which learning rates diverge?
3. What's the optimal range?

---

## Part 5: Advanced Challenges

### Exercise 5.1: Momentum ⭐⭐⭐
**Task:** Implement SGD with momentum

```python
def sgd_with_momentum(params, grads, velocities, learning_rate, momentum=0.9):
    """
    Update parameters using SGD with momentum.
    
    v_t = β·v_{t-1} + ∇L
    θ_t = θ_{t-1} - α·v_t
    
    Args:
        params: Current parameters
        grads: Gradients
        velocities: Velocity terms (updated in-place)
        learning_rate: Step size
        momentum: Momentum coefficient
    
    Returns:
        Updated params and velocities
    """
    # YOUR CODE HERE
    pass

# Compare with vanilla SGD
# Plot convergence curves
```

---

### Exercise 5.2: Adam Optimizer ⭐⭐⭐
**Task:** Implement Adam optimizer from scratch

```python
def adam_optimizer(params, grads, m, v, t, learning_rate=0.001, 
                   beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam optimizer.
    
    m_t = β₁·m_{t-1} + (1-β₁)·∇L
    v_t = β₂·v_{t-1} + (1-β₂)·∇L²
    m̂_t = m_t / (1 - β₁^t)
    v̂_t = v_t / (1 - β₂^t)
    θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)
    
    Args:
        params: Current parameters
        grads: Gradients
        m: First moment estimates (updated in-place)
        v: Second moment estimates (updated in-place)
        t: Time step
        learning_rate: Step size
        beta1: First moment decay rate
        beta2: Second moment decay rate
        epsilon: Small constant for numerical stability
    
    Returns:
        Updated params, m, v
    """
    # YOUR CODE HERE
    pass
```

---

### Exercise 5.3: Real Dataset Application ⭐⭐⭐
**Task:** Apply to MNIST digit recognition

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# One-hot encode labels
y_onehot = np.zeros((y.size, 10))
y_onehot[np.arange(y.size), y] = 1

# Split and standardize
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# YOUR TASK:
# 1. Design appropriate architecture (64 → ? → ? → 10)
# 2. Train using your implementation
# 3. Achieve >90% accuracy
# 4. Plot confusion matrix
# 5. Analyze misclassified examples

# Evaluation
def accuracy(Y_pred, Y_true):
    pred_classes = np.argmax(Y_pred, axis=1)
    true_classes = np.argmax(Y_true, axis=1)
    return np.mean(pred_classes == true_classes)

# YOUR CODE HERE
```

**Target:** >90% test accuracy

---

## Part 6: Interview-Style Questions

### Question 6.1: Conceptual ⭐
**Q:** Why do we use non-linear activation functions?

**Your answer:**
```
[Write your explanation here]
```

**Key points to cover:**
- Without non-linearity, network is just linear transformation
- Multiple linear layers = single linear layer
- Non-linearity enables learning complex functions
- Examples: XOR problem

---

### Question 6.2: Mathematical ⭐⭐
**Q:** Derive the gradient for this loss function:
```
L = -∑ᵢ yᵢ log(ŷᵢ)  (cross-entropy loss)
```

**Your derivation:**
```
∂L/∂ŷⱼ = ?

[Show your work here]
```

**Answer:** ∂L/∂ŷⱼ = -yⱼ/ŷⱼ

---

### Question 6.3: Debugging ⭐⭐⭐
**Q:** Your network's training loss is stuck at a high value. List 5 potential causes and how to diagnose each.

**Your answer:**
```
1. [Cause + diagnostic approach]
2. [Cause + diagnostic approach]
3. [Cause + diagnostic approach]
4. [Cause + diagnostic approach]
5. [Cause + diagnostic approach]
```

---

### Question 6.4: Coding Challenge ⭐⭐⭐
**Q:** Implement this function in 15 minutes:

```python
def find_optimal_learning_rate(f, grad_f, x0, lr_min=1e-5, lr_max=1.0, n_steps=50):
    """
    Find optimal learning rate using a learning rate range test.
    
    Try learning rates from lr_min to lr_max (log scale).
    For each, take n_steps and record final loss.
    Return the learning rate that gave lowest loss.
    
    Args:
        f: Function to minimize
        grad_f: Gradient function
        x0: Starting point
        lr_min: Minimum learning rate to try
        lr_max: Maximum learning rate to try
        n_steps: Steps per learning rate
    
    Returns:
        best_lr: Optimal learning rate
        results: Dictionary mapping learning rate to final loss
    """
    # YOUR CODE HERE (15 minutes)
    pass

# Test it
def test_quadratic(x):
    return np.sum((x - np.array([1, 2]))**2)

def test_quadratic_grad(x):
    return 2*(x - np.array([1, 2]))

best_lr, results = find_optimal_learning_rate(
    test_quadratic, test_quadratic_grad, 
    np.array([10.0, 10.0])
)

print(f"Best learning rate: {best_lr}")
```

---

## Solutions and Hints

### Getting Started
1. Start with easy exercises (⭐)
2. Use `optimization_tutorial.py` as reference
3. Test incrementally (don't write everything at once)
4. Use print statements for debugging
5. Compare with expected outputs

### Common Mistakes
- Forgetting to divide by batch size in gradients
- Wrong matrix dimensions in backprop
- Not initializing velocities for momentum
- Numerical instability in loss functions

### Testing Your Code
```python
# Always test with simple cases where you know the answer
def simple_test():
    # Linear function: f(x) = 2x + 1
    # Gradient should be exactly 2
    def f(x):
        return 2*x + 1
    
    def grad_f(x):
        return 2
    
    # Test your gradient descent
    x0 = 10.0
    x_opt = gradient_descent(f, grad_f, x0, lr=0.1, n_iter=100)
    
    # Should converge to some minimum
    # (though this function has no minimum, it should at least decrease)
```

---

## Next Steps

After completing these exercises:

1. ✅ **Review Solutions** - Check against `optimization_tutorial.py`
2. ✅ **Benchmark Performance** - Time your implementations
3. ✅ **Add Features** - Implement batch normalization, dropout
4. ✅ **Scale Up** - Try larger networks and datasets
5. ✅ **Learn Frameworks** - Translate knowledge to PyTorch/TensorFlow

---

## Evaluation Rubric

### For Self-Assessment

**Basic Understanding** (⭐)
- [ ] Can implement gradient descent
- [ ] Understands forward propagation
- [ ] Can compute simple derivatives

**Intermediate Skills** (⭐⭐)
- [ ] Can implement backpropagation
- [ ] Understands optimization algorithms
- [ ] Can debug training issues

**Advanced Proficiency** (⭐⭐⭐)
- [ ] Can implement modern optimizers
- [ ] Applies to real datasets
- [ ] Optimizes performance
- [ ] Explains trade-offs clearly

---

**Good luck with your practice!** 🚀

*Remember: The goal isn't to memorize code, but to deeply understand the concepts so you can implement and debug confidently in any interview or project.*
