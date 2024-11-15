# final_project
# Detailed Analysis of Data Generation Methods and Distributions

## 1. Discrete Data Generation

### Root Nodes (No Parents)

#### Visit_to_Asia
```python
data['Visit_to_Asia'] = bernoulli.rvs(np.random.uniform(0.01, 0.1), size=n_samples)
```
- Distribution: Bernoulli (Binary)
- Probability range: 1-10%
- Rationale: Rare event probability (uncommon to visit Asia)
- Values: 0 (No visit) or 1 (Visited)

#### Smoking
```python
data['Smoking'] = bernoulli.rvs(np.random.uniform(0.2, 0.5), size=n_samples)
```
- Distribution: Bernoulli (Binary)
- Probability range: 20-50%
- Rationale: Based on average smoking prevalence in population
- Values: 0 (Non-smoker) or 1 (Smoker)

### Dependent Nodes (With Parents)

#### Tuberculosis
```python
data['Tuberculosis'] = np.where(
    data['Visit_to_Asia'] == 1,
    bernoulli.rvs(np.random.uniform(0.05, 0.2), size=n_samples),  # If visited Asia
    bernoulli.rvs(np.random.uniform(0.01, 0.05), size=n_samples)  # If not visited
)
```
- Distribution: Conditional Bernoulli
- Conditions:
  * If visited Asia: 5-20% probability
  * If not visited: 1-5% probability
- Rationale: Higher TB risk in Asia visitors

#### Cancer
```python
data['Cancer'] = np.where(
    data['Smoking'] == 1,
    bernoulli.rvs(np.random.uniform(0.1, 0.3), size=n_samples),   # If smoker
    bernoulli.rvs(np.random.uniform(0.01, 0.1), size=n_samples)   # If non-smoker
)
```
- Distribution: Conditional Bernoulli
- Conditions:
  * If smoker: 10-30% probability
  * If non-smoker: 1-10% probability
- Rationale: Smoking increases cancer risk

## 2. Continuous Data Generation

### Root Nodes

#### Visit_to_Asia
```python
data['Visit_to_Asia'] = norm.rvs(loc=0, scale=1, size=n_samples)
```
- Distribution: Normal (Gaussian)
- Parameters:
  * Mean (loc) = 0
  * Standard deviation (scale) = 1
- Rationale: Standardized normal distribution

#### Smoking
```python
data['Smoking'] = norm.rvs(loc=1, scale=2, size=n_samples)
```
- Distribution: Normal (Gaussian)
- Parameters:
  * Mean (loc) = 1
  * Standard deviation (scale) = 2
- Rationale: Wider distribution to capture smoking intensity

### Dependent Nodes

#### Tuberculosis
```python
noise = np.random.normal(0, 0.1, n_samples)
data['Tuberculosis'] = 0.7 * data['Visit_to_Asia'] + noise
```
- Distribution: Linear combination + Gaussian noise
- Components:
  * Linear factor: 0.7 (strength of relationship)
  * Noise: Normal(0, 0.1)
- Rationale: Linear relationship with parent plus small random variation

## 3. Mixed Data Generation

### Continuous Variables

#### Smoking
```python
data['Smoking'] = np.where(
    data['Smoking_discrete'] == 1,
    norm.rvs(loc=1, scale=0.2, size=n_samples),  # For smokers
    norm.rvs(loc=0, scale=0.2, size=n_samples)   # For non-smokers
)
```
- Distribution: Mixture of two Normal distributions
- Parameters:
  * Smokers: Normal(1, 0.2)
  * Non-smokers: Normal(0, 0.2)
- Rationale: Separate distributions for smokers and non-smokers

#### Dyspnea
```python
# Get probabilities from logistic regression
probs = lr.predict_proba(X)[:, 1]
data['Dyspnea'] = probs + norm.rvs(loc=0, scale=0.1, size=n_samples)
```
- Distribution: Logistic transform + Gaussian noise
- Components:
  * Base: Logistic regression probabilities
  * Noise: Normal(0, 0.1)
- Rationale: Maintains relationships while providing continuous values

### Discrete Variables (Remaining)
All other variables follow the same distributions as in the discrete case.

## 4. Key Distribution Parameters Summary

### Discrete Case
| Variable      | Distribution | Parameters              | Dependencies    |
|---------------|-------------|-------------------------|-----------------|
| Visit_to_Asia | Bernoulli   | p ∈ [0.01, 0.1]        | None           |
| Smoking       | Bernoulli   | p ∈ [0.2, 0.5]         | None           |
| Tuberculosis  | Conditional | p ∈ [0.05, 0.2] or [0.01, 0.05] | Visit_to_Asia |
| Cancer        | Conditional | p ∈ [0.1, 0.3] or [0.01, 0.1]   | Smoking       |

### Continuous Case
| Variable      | Distribution | Parameters        | Dependencies    |
|---------------|-------------|-------------------|-----------------|
| Visit_to_Asia | Normal      | μ=0, σ=1         | None           |
| Smoking       | Normal      | μ=1, σ=2         | None           |
| Tuberculosis  | Linear+Noise| β=0.7, σ_noise=0.1| Visit_to_Asia |
| Cancer        | Linear+Noise| β=0.8, σ_noise=0.1| Smoking       |

### Mixed Case
| Variable      | Distribution | Parameters              | Type      |
|---------------|-------------|-------------------------|-----------|
| Smoking       | Mix Normal  | μ₁=1, μ₂=0, σ=0.2      | Continuous|
| Dyspnea       | Logistic+Noise| σ_noise=0.1          | Continuous|
| Others        | Bernoulli   | Various                 | Discrete  |

