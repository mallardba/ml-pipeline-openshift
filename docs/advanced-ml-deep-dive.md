# Advanced Machine Learning: Deep Dive Guide

## Table of Contents
1. [Mathematical Foundations](#mathematical-foundations)
2. [Advanced Algorithms](#advanced-algorithms)
3. [Deep Learning Architectures](#deep-learning-architectures)
4. [Optimization Theory](#optimization-theory)
5. [Statistical Learning Theory](#statistical-learning-theory)
6. [Advanced Topics](#advanced-topics)

---

## Mathematical Foundations

### Linear Algebra for Machine Learning

#### Vector Spaces and Subspaces
**Definition**: A vector space V over a field F is a set of vectors with two operations:
- Vector addition: v + w ∈ V
- Scalar multiplication: αv ∈ V

**Key Properties**:
- **Span**: span{v₁, v₂, ..., vₙ} = {α₁v₁ + α₂v₂ + ... + αₙvₙ | αᵢ ∈ F}
- **Linear Independence**: Vectors are linearly independent if no vector can be written as a linear combination of others
- **Basis**: A linearly independent set that spans the entire space
- **Dimension**: Number of vectors in any basis

**Applications in ML**:
```python
import numpy as np
from scipy.linalg import null_space

# Example: Finding the null space of a matrix
A = np.array([[1, 2, 3], [4, 5, 6]])
null_space_A = null_space(A)
print(f"Null space dimension: {null_space_A.shape[1]}")

# Rank and nullity theorem: rank(A) + nullity(A) = n
rank_A = np.linalg.matrix_rank(A)
nullity_A = null_space_A.shape[1]
print(f"Rank: {rank_A}, Nullity: {nullity_A}")
```

#### Eigenvalues and Eigenvectors
**Mathematical Definition**: For matrix A, if Av = λv for non-zero vector v, then λ is an eigenvalue and v is an eigenvector.

**Properties**:
- **Characteristic Polynomial**: det(A - λI) = 0
- **Spectral Decomposition**: A = VΛV⁻¹ where V contains eigenvectors and Λ contains eigenvalues
- **Trace**: tr(A) = Σλᵢ
- **Determinant**: det(A) = Πλᵢ

**Advanced Applications**:
```python
import numpy as np
from scipy.linalg import eig

# Power iteration for dominant eigenvalue
def power_iteration(A, num_iterations=100):
    b_k = np.random.rand(A.shape[0])
    for _ in range(num_iterations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    eigenvalue = np.dot(b_k, np.dot(A, b_k))
    return eigenvalue, b_k

# Example usage
A = np.array([[4, 1], [2, 3]])
eigenval, eigenvec = power_iteration(A)
print(f"Dominant eigenvalue: {eigenval}")
print(f"Corresponding eigenvector: {eigenvec}")
```

#### Matrix Decompositions

##### Singular Value Decomposition (SVD)
**Mathematical Form**: A = UΣVᵀ where:
- U: Left singular vectors (orthogonal)
- Σ: Singular values (diagonal matrix)
- V: Right singular vectors (orthogonal)

**Applications**:
- **Dimensionality Reduction**: Truncated SVD for PCA
- **Matrix Approximation**: Low-rank approximation
- **Pseudoinverse**: A⁺ = VΣ⁺Uᵀ

```python
import numpy as np
from scipy.linalg import svd

# SVD for matrix approximation
def svd_approximation(A, k):
    U, s, Vt = svd(A, full_matrices=False)
    # Keep only top k singular values
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]
    return U_k @ np.diag(s_k) @ Vt_k

# Example: Image compression using SVD
A = np.random.rand(100, 100)  # Simulated image
A_approx = svd_approximation(A, 10)  # Keep top 10 components
compression_ratio = (10 * (100 + 100 + 1)) / (100 * 100)
print(f"Compression ratio: {compression_ratio:.2%}")
```

##### QR Decomposition
**Mathematical Form**: A = QR where:
- Q: Orthogonal matrix
- R: Upper triangular matrix

**Applications**:
- **Least Squares**: Solving Ax = b becomes Rx = Qᵀb
- **Eigenvalue Computation**: QR algorithm
- **Orthogonalization**: Gram-Schmidt process

```python
import numpy as np
from scipy.linalg import qr

# QR decomposition for least squares
def qr_least_squares(A, b):
    Q, R = qr(A, mode='economic')
    # Solve Rx = Q^T b
    y = Q.T @ b
    x = np.linalg.solve(R, y)
    return x

# Example
A = np.random.rand(50, 10)
b = np.random.rand(50)
x = qr_least_squares(A, b)
print(f"Solution: {x}")
```

### Probability Theory

#### Advanced Probability Distributions

##### Multivariate Normal Distribution
**PDF**: f(x) = (2π)^(-k/2) |Σ|^(-1/2) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))

**Properties**:
- **Marginal Distributions**: Also normal
- **Conditional Distributions**: Normal with updated parameters
- **Linear Transformations**: Preserve normality

```python
import numpy as np
from scipy.stats import multivariate_normal

# Multivariate normal sampling and conditioning
def conditional_normal(mu, Sigma, x_obs, obs_indices):
    """Compute conditional distribution given observed values"""
    n = len(mu)
    unobs_indices = [i for i in range(n) if i not in obs_indices]
    
    # Partition mean and covariance
    mu_1 = mu[obs_indices]
    mu_2 = mu[unobs_indices]
    
    Sigma_11 = Sigma[np.ix_(obs_indices, obs_indices)]
    Sigma_12 = Sigma[np.ix_(obs_indices, unobs_indices)]
    Sigma_21 = Sigma[np.ix_(unobs_indices, obs_indices)]
    Sigma_22 = Sigma[np.ix_(unobs_indices, unobs_indices)]
    
    # Conditional parameters
    mu_cond = mu_2 + Sigma_21 @ np.linalg.inv(Sigma_11) @ (x_obs - mu_1)
    Sigma_cond = Sigma_22 - Sigma_21 @ np.linalg.inv(Sigma_11) @ Sigma_12
    
    return mu_cond, Sigma_cond

# Example usage
mu = np.array([0, 0, 0])
Sigma = np.array([[1, 0.5, 0.2], [0.5, 1, 0.3], [0.2, 0.3, 1]])
x_obs = np.array([1.0])  # Observed first component
obs_indices = [0]

mu_cond, Sigma_cond = conditional_normal(mu, Sigma, x_obs, obs_indices)
print(f"Conditional mean: {mu_cond}")
print(f"Conditional covariance: {Sigma_cond}")
```

##### Exponential Family Distributions
**General Form**: p(x|η) = h(x) exp(ηᵀT(x) - A(η))

**Key Members**:
- **Gaussian**: η = (μ/σ², -1/(2σ²)), T(x) = (x, x²)
- **Bernoulli**: η = log(p/(1-p)), T(x) = x
- **Poisson**: η = log(λ), T(x) = x

**Properties**:
- **Sufficient Statistics**: T(x) contains all information about η
- **Conjugate Priors**: Natural conjugate priors exist
- **Maximum Likelihood**: Closed-form solutions often available

```python
import numpy as np
from scipy.special import logsumexp

# Exponential family maximum likelihood
def exponential_family_mle(T_samples, natural_params_init):
    """Find MLE for exponential family distribution"""
    def neg_log_likelihood(natural_params):
        # A(η) = log ∫ h(x) exp(ηᵀT(x)) dx
        # For most distributions, this has closed form
        eta_sum = np.sum(T_samples @ natural_params)
        log_A = logsumexp(T_samples @ natural_params)
        return -(eta_sum - len(T_samples) * log_A)
    
    from scipy.optimize import minimize
    result = minimize(neg_log_likelihood, natural_params_init)
    return result.x

# Example: Bernoulli MLE
n_samples = 1000
p_true = 0.3
samples = np.random.binomial(1, p_true, n_samples)
T_samples = samples.reshape(-1, 1)  # Sufficient statistic

eta_mle = exponential_family_mle(T_samples, np.array([0.0]))
p_mle = 1 / (1 + np.exp(-eta_mle[0]))  # Convert back to probability
print(f"True p: {p_true}, MLE p: {p_mle}")
```

#### Information Theory

##### Entropy and Mutual Information
**Entropy**: H(X) = -Σ p(x) log p(x)
**Mutual Information**: I(X;Y) = H(X) + H(Y) - H(X,Y)

**Applications**:
- **Feature Selection**: High mutual information with target
- **Clustering**: Maximize intra-cluster similarity
- **Dimensionality Reduction**: Preserve mutual information

```python
import numpy as np
from scipy.stats import entropy

def mutual_information(X, Y, bins=10):
    """Estimate mutual information between continuous variables"""
    # Discretize continuous variables
    X_discrete = np.digitize(X, np.linspace(X.min(), X.max(), bins))
    Y_discrete = np.digitize(Y, np.linspace(Y.min(), Y.max(), bins))
    
    # Joint and marginal distributions
    joint_hist, _, _ = np.histogram2d(X_discrete, Y_discrete, bins=bins)
    joint_prob = joint_hist / joint_hist.sum()
    
    X_prob = joint_prob.sum(axis=1)
    Y_prob = joint_prob.sum(axis=0)
    
    # Mutual information
    mi = 0
    for i in range(bins):
        for j in range(bins):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (X_prob[i] * Y_prob[j]))
    
    return mi

# Example
X = np.random.normal(0, 1, 1000)
Y = X + np.random.normal(0, 0.5, 1000)  # Y correlated with X
mi = mutual_information(X, Y)
print(f"Mutual Information: {mi:.3f}")
```

---

## Advanced Algorithms

### Ensemble Methods Deep Dive

#### Boosting Theory

##### AdaBoost Algorithm
**Mathematical Foundation**:
1. Initialize weights: w₁ᵢ = 1/N
2. For t = 1 to T:
   - Train weak learner hₜ with weights wₜ
   - Compute error: εₜ = Σᵢ wₜᵢ I(yᵢ ≠ hₜ(xᵢ))
   - Compute weight: αₜ = ½ log((1-εₜ)/εₜ)
   - Update weights: wₜ₊₁ᵢ = wₜᵢ exp(-αₜ yᵢ hₜ(xᵢ))
3. Final classifier: H(x) = sign(Σₜ αₜ hₜ(x))

**Theoretical Guarantees**:
- **Training Error Bound**: P(H(x) ≠ y) ≤ exp(-2 Σₜ (½ - εₜ)²)
- **Generalization Bound**: With probability 1-δ, generalization error ≤ training error + O(√(T log N)/N)

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples
        
        for t in range(self.n_estimators):
            # Train weak learner
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.fit(X, y, sample_weight=weights)
            
            # Compute error and weight
            predictions = estimator.predict(X)
            error = np.sum(weights * (predictions != y))
            
            if error >= 0.5:  # Stop if error too high
                break
                
            alpha = 0.5 * np.log((1 - error) / error)
            
            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  # Normalize
            
            self.estimators.append(estimator)
            self.estimator_weights.append(alpha)
    
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for estimator, alpha in zip(self.estimators, self.estimator_weights):
            predictions += alpha * estimator.predict(X)
        return np.sign(predictions)
```

##### Gradient Boosting
**Mathematical Foundation**:
- **Additive Model**: F(x) = Σₘ fₘ(x)
- **Gradient Descent**: fₘ(x) = -ρₘ ∇Fₘ₋₁ L(Fₘ₋₁(xᵢ), yᵢ)
- **Line Search**: ρₘ = argmin_ρ L(Fₘ₋₁ + ρfₘ)

**Advanced Variants**:
- **XGBoost**: Regularized gradient boosting with second-order approximation
- **LightGBM**: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB)
- **CatBoost**: Ordered boosting and categorical feature handling

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators = []
    
    def fit(self, X, y):
        # Initialize with mean
        self.mean = np.mean(y)
        F = np.full(len(y), self.mean)
        
        for m in range(self.n_estimators):
            # Compute negative gradient (residuals)
            residuals = y - F
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # Update model
            F += self.learning_rate * tree.predict(X)
            self.estimators.append(tree)
    
    def predict(self, X):
        predictions = np.full(X.shape[0], self.mean)
        for tree in self.estimators:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
```

#### Stacking and Blending

##### Stacking Theory
**Mathematical Formulation**:
- **Level 1**: Train base learners h₁, h₂, ..., hₘ
- **Level 2**: Train meta-learner g on predictions h₁(x), h₂(x), ..., hₘ(x)
- **Final Prediction**: ŷ = g(h₁(x), h₂(x), ..., hₘ(x))

**Cross-Validation Stacking**:
```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

class StackingRegressor:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.trained_base_models = []
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        
        # Create out-of-fold predictions
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in kf.split(X):
                model.fit(X[train_idx], y[train_idx])
                meta_features[val_idx, i] = model.predict(X[val_idx])
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)
        
        # Retrain base models on full data
        for model in self.base_models:
            model.fit(X, y)
            self.trained_base_models.append(model)
    
    def predict(self, X):
        meta_features = np.zeros((X.shape[0], len(self.trained_base_models)))
        for i, model in enumerate(self.trained_base_models):
            meta_features[:, i] = model.predict(X)
        return self.meta_model.predict(meta_features)
```

---

## Deep Learning Architectures

### Advanced Neural Network Architectures

#### Residual Networks (ResNet)
**Mathematical Foundation**:
- **Residual Block**: y = F(x, {Wᵢ}) + x
- **Identity Mapping**: Allows gradients to flow directly
- **Skip Connections**: Prevent vanishing gradient problem

**Advanced Variants**:
- **DenseNet**: Dense connections between all layers
- **ResNeXt**: Grouped convolutions for efficiency
- **EfficientNet**: Compound scaling of depth, width, and resolution

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample:
            residual = self.downsample(x)
        
        out += residual
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
```

#### Attention Mechanisms

##### Self-Attention
**Mathematical Formulation**:
- **Query, Key, Value**: Q = XW_Q, K = XW_K, V = XW_V
- **Attention Weights**: A = softmax(QK^T / √d_k)
- **Output**: O = AV

**Multi-Head Attention**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(attention_output)
        
        return output, attention_weights
```

#### Transformer Architecture
**Complete Transformer Implementation**:
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0)
        
        # Embeddings
        x = self.token_embedding(x) * np.sqrt(self.d_model)
        x += self.position_embedding(positions)
        x = self.dropout(x)
        
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        return self.layer_norm(x)
```

---

## Optimization Theory

### Advanced Optimization Algorithms

#### Convex Optimization
**Convex Function Properties**:
- **Jensen's Inequality**: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
- **First-Order Condition**: f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)
- **Second-Order Condition**: ∇²f(x) ⪰ 0 (positive semidefinite)

**Convex Optimization Problems**:
```python
import numpy as np
from scipy.optimize import minimize
import cvxpy as cp

# Example: Portfolio Optimization (Markowitz)
def markowitz_portfolio(returns, risk_aversion=1.0):
    """Solve Markowitz portfolio optimization problem"""
    n_assets = returns.shape[1]
    mu = np.mean(returns, axis=0)  # Expected returns
    Sigma = np.cov(returns.T)     # Covariance matrix
    
    # Variables
    w = cp.Variable(n_assets)
    
    # Objective: maximize return - risk_aversion * variance
    objective = cp.Maximize(mu.T @ w - risk_aversion * cp.quad_form(w, Sigma))
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,  # Weights sum to 1
        w >= 0           # Long-only portfolio
    ]
    
    # Solve
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return w.value

# Example usage
np.random.seed(42)
returns = np.random.multivariate_normal([0.05, 0.08, 0.12], 
                                       [[0.1, 0.02, 0.01],
                                        [0.02, 0.15, 0.03],
                                        [0.01, 0.03, 0.2]], 1000)
weights = markowitz_portfolio(returns, risk_aversion=2.0)
print(f"Optimal weights: {weights}")
```

#### Non-Convex Optimization

##### Simulated Annealing
**Algorithm**:
1. Start with initial solution x₀ and temperature T₀
2. For each iteration:
   - Generate candidate x' = x + random perturbation
   - Accept with probability min(1, exp(-(f(x')-f(x))/T))
   - Decrease temperature T
3. Return best solution found

```python
import numpy as np
import math

def simulated_annealing(objective, bounds, max_iterations=1000, initial_temp=100):
    """Simulated annealing for global optimization"""
    # Initialize
    best_solution = np.random.uniform(bounds[0], bounds[1])
    best_value = objective(best_solution)
    current_solution = best_solution.copy()
    current_value = best_value
    
    temperature = initial_temp
    
    for iteration in range(max_iterations):
        # Generate candidate solution
        candidate = current_solution + np.random.normal(0, 0.1, size=current_solution.shape)
        candidate = np.clip(candidate, bounds[0], bounds[1])
        
        # Evaluate candidate
        candidate_value = objective(candidate)
        
        # Accept or reject
        if candidate_value < current_value:
            # Always accept better solutions
            current_solution = candidate
            current_value = candidate_value
            
            if candidate_value < best_value:
                best_solution = candidate
                best_value = candidate_value
        else:
            # Accept worse solutions with probability
            probability = math.exp(-(candidate_value - current_value) / temperature)
            if np.random.random() < probability:
                current_solution = candidate
                current_value = candidate_value
        
        # Cool down
        temperature *= 0.95
    
    return best_solution, best_value

# Example: Minimize Rosenbrock function
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

bounds = [-2, 2]
solution, value = simulated_annealing(rosenbrock, bounds)
print(f"Solution: {solution}, Value: {value}")
```

##### Genetic Algorithms
```python
import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.8):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def initialize_population(self, bounds, chromosome_length):
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            chromosome = np.random.uniform(bounds[0], bounds[1], chromosome_length)
            population.append(chromosome)
        return population
    
    def evaluate_fitness(self, population, objective):
        """Evaluate fitness of each individual"""
        fitness = []
        for individual in population:
            fitness.append(-objective(individual))  # Minimize objective
        return np.array(fitness)
    
    def selection(self, population, fitness):
        """Tournament selection"""
        selected = []
        for _ in range(self.population_size):
            # Tournament of size 3
            tournament = random.sample(range(len(population)), 3)
            winner = max(tournament, key=lambda i: fitness[i])
            selected.append(population[winner].copy())
        return selected
    
    def crossover(self, parent1, parent2):
        """Uniform crossover"""
        if random.random() < self.crossover_rate:
            child1 = parent1.copy()
            child2 = parent2.copy()
            
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    child1[i], child2[i] = child2[i], child1[i]
            
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
    
    def mutation(self, individual, bounds):
        """Gaussian mutation"""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] += np.random.normal(0, 0.1)
                mutated[i] = np.clip(mutated[i], bounds[0], bounds[1])
        return mutated
    
    def evolve(self, objective, bounds, chromosome_length, generations=100):
        """Main evolution loop"""
        population = self.initialize_population(bounds, chromosome_length)
        
        for generation in range(generations):
            fitness = self.evaluate_fitness(population, objective)
            
            # Selection
            selected = self.selection(population, fitness)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, self.population_size, 2):
                child1, child2 = self.crossover(selected[i], selected[i+1])
                child1 = self.mutation(child1, bounds)
                child2 = self.mutation(child2, bounds)
                new_population.extend([child1, child2])
            
            population = new_population
            
            # Track best solution
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness}")
        
        # Return best solution
        final_fitness = self.evaluate_fitness(population, objective)
        best_idx = np.argmax(final_fitness)
        return population[best_idx], final_fitness[best_idx]

# Example usage
ga = GeneticAlgorithm()
solution, fitness = ga.evolve(rosenbrock, bounds, chromosome_length=2)
print(f"GA Solution: {solution}, Fitness: {fitness}")
```

---

## Statistical Learning Theory

### PAC Learning Theory
**Probably Approximately Correct (PAC) Learning**:
- **Definition**: A concept class C is PAC-learnable if there exists an algorithm that, given ε > 0 and δ > 0, outputs a hypothesis h such that P[error(h) ≤ ε] ≥ 1 - δ
- **Sample Complexity**: Number of examples needed for PAC learning
- **VC Dimension**: Measure of model complexity

```python
import numpy as np
from scipy.stats import binom

def pac_sample_complexity(epsilon, delta, vc_dimension):
    """Calculate PAC sample complexity"""
    # Upper bound on sample complexity
    m = (8 / epsilon) * (vc_dimension * np.log(13/epsilon) + np.log(2/delta))
    return int(np.ceil(m))

def vc_dimension_bound(error_rate, sample_size, confidence=0.95):
    """VC dimension bound on generalization error"""
    delta = 1 - confidence
    vc_bound = np.sqrt((np.log(2 * sample_size) + 1) / sample_size) + np.sqrt(np.log(1/delta) / (2 * sample_size))
    return error_rate + vc_bound

# Example: VC dimension for linear classifiers in 2D
def linear_classifier_vc_dimension(dimension):
    """VC dimension of linear classifiers in d dimensions"""
    return dimension + 1

# Calculate sample complexity for 2D linear classifier
epsilon = 0.1
delta = 0.05
vc_dim = linear_classifier_vc_dimension(2)
sample_complexity = pac_sample_complexity(epsilon, delta, vc_dim)
print(f"Sample complexity for ε={epsilon}, δ={delta}: {sample_complexity}")
```

### Bias-Variance Decomposition
**Mathematical Formulation**:
E[(y - ŷ)²] = Bias²(ŷ) + Var(ŷ) + σ²

Where:
- **Bias**: E[ŷ] - y (systematic error)
- **Variance**: E[(ŷ - E[ŷ])²] (sensitivity to training data)
- **Irreducible Error**: σ² (noise in data)

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def bias_variance_decomposition(X, y, model, n_bootstrap=100):
    """Estimate bias and variance of a model"""
    n_samples = X.shape[0]
    predictions = np.zeros((n_bootstrap, n_samples))
    
    # Bootstrap sampling
    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]
        y_boot = y[indices]
        
        model.fit(X_boot, y_boot)
        predictions[i] = model.predict(X)
    
    # Calculate bias and variance
    mean_predictions = np.mean(predictions, axis=0)
    bias_squared = np.mean((mean_predictions - y) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    
    return bias_squared, variance

# Example: Compare different models
np.random.seed(42)
X = np.random.randn(100, 5)
y = X @ np.random.randn(5) + 0.1 * np.random.randn(100)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(max_depth=3),
    'Random Forest': RandomForestRegressor(n_estimators=10, max_depth=3)
}

for name, model in models.items():
    bias_sq, variance = bias_variance_decomposition(X, y, model)
    total_error = bias_sq + variance
    print(f"{name}: Bias²={bias_sq:.4f}, Variance={variance:.4f}, Total={total_error:.4f}")
```

---

## Advanced Topics

### Causal Inference
**Causal vs Statistical Relationships**:
- **Correlation**: Statistical association between variables
- **Causation**: One variable directly influences another
- **Confounding**: Hidden variables affecting both cause and effect

**Methods**:
- **Randomized Controlled Trials**: Gold standard for causal inference
- **Instrumental Variables**: Use external variation to identify causal effects
- **Regression Discontinuity**: Exploit arbitrary thresholds
- **Difference-in-Differences**: Compare changes over time

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def instrumental_variables_2sls(X, Z, y):
    """Two-stage least squares for instrumental variables"""
    # First stage: X = Z * gamma + error
    first_stage = LinearRegression()
    first_stage.fit(Z, X)
    X_predicted = first_stage.predict(Z)
    
    # Second stage: y = X_predicted * beta + error
    second_stage = LinearRegression()
    second_stage.fit(X_predicted.reshape(-1, 1), y)
    
    return second_stage.coef_[0], second_stage.intercept_

# Example: Education and earnings with distance to college as instrument
np.random.seed(42)
n = 1000

# True causal effect of education on earnings
true_effect = 0.5

# Instrument: distance to nearest college (affects education but not earnings directly)
distance = np.random.exponential(2, n)
education = 12 + 4 * np.exp(-distance) + np.random.normal(0, 1, n)

# Earnings: affected by education and unobserved ability
ability = np.random.normal(0, 1, n)
earnings = 20 + true_effect * education + 2 * ability + np.random.normal(0, 2, n)

# OLS (biased due to omitted variable bias)
ols = LinearRegression()
ols.fit(education.reshape(-1, 1), earnings)
ols_effect = ols.coef_[0]

# IV estimation
iv_effect, iv_intercept = instrumental_variables_2sls(education, distance, earnings)

print(f"True effect: {true_effect:.3f}")
print(f"OLS estimate: {ols_effect:.3f} (biased)")
print(f"IV estimate: {iv_effect:.3f} (unbiased)")
```

### Reinforcement Learning
**Markov Decision Process (MDP)**:
- **States**: S (set of possible states)
- **Actions**: A (set of possible actions)
- **Transition Probabilities**: P(s'|s,a)
- **Rewards**: R(s,a,s')
- **Policy**: π(a|s) (probability of taking action a in state s)

**Value Functions**:
- **State Value**: V^π(s) = E[Σₜ γᵗ Rₜ | s₀ = s, π]
- **Action Value**: Q^π(s,a) = E[Σₜ γᵗ Rₜ | s₀ = s, a₀ = a, π]

```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))
    
    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        """Q-learning update rule"""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state, action] = new_q
    
    def train(self, env, episodes=1000):
        """Train the agent"""
        for episode in range(episodes):
            state = env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state)
                state = next_state
            
            # Decay epsilon
            if episode % 100 == 0:
                self.epsilon *= 0.99

# Example: Simple Grid World
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = 0
        self.goal = size * size - 1
        
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        # Actions: 0=up, 1=right, 2=down, 3=left
        row, col = self.state // self.size, self.state % self.size
        
        if action == 0 and row > 0:  # Up
            self.state -= self.size
        elif action == 1 and col < self.size - 1:  # Right
            self.state += 1
        elif action == 2 and row < self.size - 1:  # Down
            self.state += self.size
        elif action == 3 and col > 0:  # Left
            self.state -= 1
        
        # Reward: -1 for each step, +10 for reaching goal
        if self.state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False
        
        return self.state, reward, done

# Train Q-learning agent
env = GridWorld()
agent = QLearning(n_states=25, n_actions=4)
agent.train(env, episodes=1000)

# Test learned policy
state = env.reset()
print("Learned path:")
while True:
    action = agent.choose_action(state, training=False)
    next_state, reward, done = env.step(action)
    print(f"State {state} -> Action {action} -> State {next_state}")
    state = next_state
    if done:
        break
```

This advanced guide provides deep mathematical foundations, sophisticated algorithms, and cutting-edge techniques that build upon the crash course. It covers theoretical aspects, practical implementations, and advanced applications that are essential for understanding modern machine learning at a professional level.
