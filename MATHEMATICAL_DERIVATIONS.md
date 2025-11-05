# Mathematical Derivations for Optimization Algorithms
## Detailed Proofs and Intuitions

---

## Table of Contents
1. [Gradient Descent Derivation](#gradient-descent-derivation)
2. [Backpropagation Algorithm](#backpropagation-algorithm)
3. [Matrix Calculus](#matrix-calculus)
4. [Chain Rule for Neural Networks](#chain-rule-for-neural-networks)
5. [Convergence Analysis](#convergence-analysis)

---

## 1. Gradient Descent Derivation

### 1.1 First-Order Taylor Expansion

For a differentiable function L(θ), we can approximate it around point θ₀:

```
L(θ₀ + Δθ) ≈ L(θ₀) + ∇L(θ₀)ᵀ·Δθ
```

**Goal**: Choose Δθ to minimize L(θ₀ + Δθ)

### 1.2 Finding the Descent Direction

To find the direction that decreases L the most, consider:

```
L(θ₀ + αd) ≈ L(θ₀) + α·∇L(θ₀)ᵀ·d
```

Where:
- α > 0 is step size
- d is unit direction vector (||d|| = 1)

**Minimize**: ∇L(θ₀)ᵀ·d

Using Cauchy-Schwarz inequality:
```
∇L(θ₀)ᵀ·d ≥ -||∇L(θ₀)|| · ||d|| = -||∇L(θ₀)||
```

Equality holds when:
```
d = -∇L(θ₀) / ||∇L(θ₀)||
```

**Therefore**, the steepest descent direction is **-∇L(θ₀)**.

### 1.3 Update Rule

```
θ_{t+1} = θ_t - α·∇L(θ_t)
```

Where α is the learning rate (step size).

### 1.4 Intuition

Think of standing on a mountain:
- ∇L(θ) points uphill (direction of steepest ascent)
- -∇L(θ) points downhill (direction of steepest descent)
- We take steps downhill to reach the valley (minimum)

---

## 2. Backpropagation Algorithm

### 2.1 Problem Setup

Given:
- Neural network: f(x; θ)
- Loss function: L = ||f(x; θ) - y||²
- Goal: Compute ∇_θ L efficiently

### 2.2 Forward Pass Equations

For a 4-layer network:

```
Layer 0 (Input):
h₀ = x

Layer 1:
z₁ = W₁·h₀ + b₁
h₁ = tanh(z₁)

Layer 2:
z₂ = W₂·h₁ + b₂
h₂ = tanh(z₂)

Layer 3:
z₃ = W₃·h₂ + b₃
h₃ = tanh(z₃)

Layer 4 (Output):
z₄ = W₄·h₃ + b₄
ŷ = z₄
```

### 2.3 Loss Function

```
L = (1/2)||ŷ - y||² = (1/2)Σᵢ(ŷᵢ - yᵢ)²
```

### 2.4 Backward Pass - Output Layer

Compute ∂L/∂z₄:

```
∂L/∂z₄ = ∂L/∂ŷ · ∂ŷ/∂z₄
       = (ŷ - y) · I
       = ŷ - y

Define: δ₄ = ŷ - y
```

Gradients for output layer parameters:

```
∂L/∂W₄ = δ₄ ⊗ h₃ᵀ = δ₄·h₃ᵀ
∂L/∂b₄ = δ₄
```

**Matrix dimensions**:
- δ₄: (n_out, 1)
- h₃: (n₃, 1)
- ∂L/∂W₄: (n_out, n₃)

### 2.5 Backward Pass - Hidden Layer 3

Compute ∂L/∂z₃:

```
∂L/∂z₃ = ∂L/∂z₄ · ∂z₄/∂h₃ · ∂h₃/∂z₃
       = δ₄ · W₄ · tanh'(z₃)

Since tanh'(z) = 1 - tanh²(z) = 1 - h₃²:

δ₃ = (W₄ᵀ·δ₄) ⊙ (1 - h₃²)
```

Where ⊙ denotes element-wise multiplication.

Gradients:
```
∂L/∂W₃ = δ₃·h₂ᵀ
∂L/∂b₃ = δ₃
```

### 2.6 General Backward Pass Formula

For any layer ℓ:

```
δℓ = (W^T_{ℓ+1}·δ_{ℓ+1}) ⊙ f'(zℓ)

∂L/∂Wℓ = δℓ·h^T_{ℓ-1}
∂L/∂bℓ = δℓ
```

This is the **chain rule** in action!

### 2.7 Complete Backpropagation Algorithm

```
1. Forward Pass:
   - Compute all z and h values from input to output
   - Store them for backward pass

2. Compute Output Error:
   δ_L = ∂L/∂z_L (depends on loss function)

3. Backward Pass (for ℓ = L-1 down to 1):
   δℓ = (W^T_{ℓ+1}·δ_{ℓ+1}) ⊙ f'(zℓ)
   
4. Compute Gradients (for all layers):
   ∇Wℓ = δℓ·h^T_{ℓ-1}
   ∇bℓ = δℓ

5. Update Parameters:
   Wℓ ← Wℓ - α·∇Wℓ
   bℓ ← bℓ - α·∇bℓ
```

---

## 3. Matrix Calculus

### 3.1 Derivative of Vector-Vector Product

Given:
```
f(x) = aᵀx
```

Then:
```
∂f/∂x = a
```

### 3.2 Derivative of Quadratic Form

Given:
```
f(x) = xᵀAx
```

Then:
```
∂f/∂x = (A + Aᵀ)x
```

If A is symmetric:
```
∂f/∂x = 2Ax
```

### 3.3 Derivative of Matrix Product

Given:
```
f(W) = trace(WᵀAWB)
```

Then:
```
∂f/∂W = AWB + AᵀWBᵀ
```

### 3.4 Practical Example: MSE Loss

Given:
```
L = ||Wx - y||²
```

Expand:
```
L = (Wx - y)ᵀ(Wx - y)
  = xᵀWᵀWx - 2yᵀWx + yᵀy
```

Compute gradient:
```
∂L/∂W = 2Wxxᵀ - 2yxᵀ
       = 2(Wx - y)xᵀ
       = 2·error·xᵀ
```

This matches our backpropagation formula!

---

## 4. Chain Rule for Neural Networks

### 4.1 Scalar Chain Rule

For y = f(g(x)):
```
dy/dx = (dy/dg) · (dg/dx)
```

### 4.2 Multivariable Chain Rule

For y = f(u₁, u₂, ..., uₙ) where each uᵢ = gᵢ(x):

```
∂y/∂x = Σᵢ (∂y/∂uᵢ) · (∂uᵢ/∂x)
```

### 4.3 Neural Network Example

Consider:
```
h₁ = tanh(W₁x + b₁)
h₂ = tanh(W₂h₁ + b₂)
L = ||h₂ - y||²
```

To find ∂L/∂W₁:

```
∂L/∂W₁ = (∂L/∂h₂) · (∂h₂/∂h₁) · (∂h₁/∂W₁)
```

Step by step:

**Step 1**: ∂L/∂h₂
```
∂L/∂h₂ = 2(h₂ - y)
```

**Step 2**: ∂h₂/∂h₁
```
h₂ = tanh(W₂h₁ + b₂)

∂h₂/∂h₁ = tanh'(W₂h₁ + b₂) · W₂
        = (1 - h₂²) · W₂
```

**Step 3**: ∂h₁/∂W₁
```
h₁ = tanh(W₁x + b₁)

∂h₁/∂W₁ = tanh'(W₁x + b₁) ⊗ x
        = (1 - h₁²) ⊗ x
```

**Combine**:
```
∂L/∂W₁ = [(2(h₂ - y)) · ((1 - h₂²) · W₂)] ⊗ [(1 - h₁²) ⊗ x]
```

This is exactly what backpropagation computes efficiently!

---

## 5. Convergence Analysis

### 5.1 Convex Optimization

For convex functions (bowl-shaped), gradient descent guarantees convergence to global minimum.

**Definition**: Function f is convex if:
```
f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)  for all λ ∈ [0,1]
```

**Property**: For convex f with L-Lipschitz gradient:
```
||∇f(x) - ∇f(y)|| ≤ L||x - y||
```

### 5.2 Convergence Rate

With learning rate α = 1/L:

```
f(x_t) - f(x*) ≤ (2L||x₀ - x*||²) / t
```

This is **O(1/t)** convergence rate.

### 5.3 Non-Convex Case (Neural Networks)

Neural networks are **non-convex**:
- Multiple local minima
- Saddle points
- No global convergence guarantee

**But in practice**:
- SGD with momentum often finds good solutions
- Over-parameterization helps (more parameters than data)
- Initialization matters (Xavier, He initialization)

### 5.4 Learning Rate Selection

**Too large**: Divergence
```
θ_{t+1} = θ_t - α·∇L(θ_t)

If α > 2/L, the update overshoots and oscillates
```

**Too small**: Slow convergence
```
Takes O(1/α) iterations to reach minimum
```

**Optimal**: α ∈ [1/L, 2/L]

**Adaptive methods** (Adam, RMSprop):
- Adjust α per parameter
- Use gradient history
- More robust to poor initialization

---

## 6. Advanced Topics

### 6.1 Second-Order Methods

Newton's method uses curvature information:

```
θ_{t+1} = θ_t - H⁻¹·∇L(θ_t)
```

Where H is the Hessian matrix:
```
H = ∇²L(θ) = [∂²L/∂θᵢ∂θⱼ]
```

**Pros**: Faster convergence (quadratic)
**Cons**: Computing H⁻¹ is O(n³), impractical for large n

### 6.2 Quasi-Newton Methods

Approximate H⁻¹ without computing it:
- BFGS
- L-BFGS (limited memory)

Used in traditional ML but not deep learning (too many parameters).

### 6.3 Momentum

SGD with momentum:

```
v_{t+1} = β·v_t + ∇L(θ_t)
θ_{t+1} = θ_t - α·v_{t+1}
```

**Intuition**: Ball rolling downhill
- Accumulates velocity
- Reduces oscillation
- Faster convergence

### 6.4 Adam Optimizer

Combines momentum with adaptive learning rates:

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t          # First moment
v_t = β₂·v_{t-1} + (1-β₂)·g_t²         # Second moment

m̂_t = m_t / (1 - β₁ᵗ)                   # Bias correction
v̂_t = v_t / (1 - β₂ᵗ)                   # Bias correction

θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)
```

**Default hyperparameters**:
- α = 0.001
- β₁ = 0.9
- β₂ = 0.999
- ε = 10⁻⁸

---

## 7. Practical Considerations

### 7.1 Gradient Checking

Verify backpropagation implementation:

```python
def gradient_check(f, x, epsilon=1e-5):
    """
    Compare analytical gradient with numerical gradient.
    """
    analytical_grad = compute_gradient(f, x)
    
    numerical_grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += epsilon
        x_minus = x.copy()
        x_minus[i] -= epsilon
        
        numerical_grad[i] = (f(x_plus) - f(x_minus)) / (2*epsilon)
    
    relative_error = np.abs(analytical_grad - numerical_grad) / \
                     (np.abs(analytical_grad) + np.abs(numerical_grad))
    
    if relative_error < 1e-7:
        print("✓ Gradient check passed!")
    else:
        print("✗ Gradient check failed!")
```

### 7.2 Numerical Stability

**Problem**: Exponentials can overflow/underflow

**Solution**: Use numerically stable implementations:

```python
# Bad: Direct computation
def softmax_bad(x):
    return np.exp(x) / np.sum(np.exp(x))

# Good: Subtract max for stability
def softmax_good(x):
    x_shifted = x - np.max(x)
    return np.exp(x_shifted) / np.sum(np.exp(x_shifted))
```

### 7.3 Vanishing Gradients

**Problem**: In deep networks, gradients become exponentially small.

**Why**: Repeated multiplication by W and σ'(z):
```
∂L/∂W₁ ∝ σ'(z₁)·W₂·σ'(z₂)·W₃·...·W_L

If σ'(z) < 1 and ||W|| < 1, product vanishes
```

**Solutions**:
1. Use ReLU instead of sigmoid/tanh
2. Batch normalization
3. Residual connections (skip connections)
4. Careful initialization

### 7.4 Exploding Gradients

**Problem**: Gradients become exponentially large.

**Why**: ||W|| > 1 and repeated multiplication

**Solutions**:
1. Gradient clipping
2. Weight regularization
3. Proper initialization

---

## 8. Summary of Key Formulas

### Gradient Descent
```
θ_{t+1} = θ_t - α·∇L(θ_t)
```

### Backpropagation
```
δ_L = ∂L/∂z_L
δℓ = (W^T_{ℓ+1}·δ_{ℓ+1}) ⊙ σ'(zℓ)
∇Wℓ = δℓ·h^T_{ℓ-1}
```

### Activation Functions
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
tanh'(z) = 1 - tanh²(z)

ReLU(z) = max(0, z)
ReLU'(z) = 1 if z > 0 else 0

sigmoid(z) = 1 / (1 + e^(-z))
sigmoid'(z) = sigmoid(z)·(1 - sigmoid(z))
```

### Loss Functions
```
MSE: L = (1/2n)Σ||ŷᵢ - yᵢ||²
∂L/∂ŷ = (1/n)(ŷ - y)

Cross-entropy: L = -Σyᵢ log(ŷᵢ)
∂L/∂ŷ = -(y/ŷ)
```

---

## 9. Further Reading

### Books
1. "Deep Learning" - Goodfellow, Bengio, Courville (Chapter 6-8)
2. "Neural Networks for Pattern Recognition" - Christopher Bishop
3. "Convex Optimization" - Boyd & Vandenberghe

### Papers
1. "Backpropagation Applied to Handwritten Zip Code Recognition" - LeCun (1989)
2. "Understanding the difficulty of training deep feedforward neural networks" - Glorot & Bengio (2010)
3. "Adam: A Method for Stochastic Optimization" - Kingma & Ba (2014)

### Online Resources
1. Stanford CS231n: http://cs231n.stanford.edu
2. Calculus on Computational Graphs: http://colah.github.io
3. Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com

---

**Remember**: Understanding these mathematical foundations deeply will set you apart in interviews and enable you to debug and improve ML systems effectively!
