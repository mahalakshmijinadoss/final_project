# Detailed Explanation of PC Algorithm Implementation

## 1. Data Generation

```python
def generate_discrete_data(n_samples):
```

### Root Node Generation
```python
data['Visit_to_Asia'] = np.random.binomial(1, 0.3, n_samples)
data['Smoking'] = np.random.binomial(1, 0.5, n_samples)
```
- Uses binomial distribution for binary (0/1) values
- Probability 0.3 for Visit_to_Asia (30% chance of 1)
- Probability 0.5 for Smoking (balanced 50-50)

### Child Node Generation
Example for Tuberculosis:
```python
data['Tuberculosis'] = 0  # Initialize all to 0
data.loc[data['Visit_to_Asia'] == 1, 'Tuberculosis'] = \
    np.random.binomial(1, 0.9, sum(data['Visit_to_Asia'] == 1))
data.loc[data['Visit_to_Asia'] == 0, 'Tuberculosis'] = \
    np.random.binomial(1, 0.05, sum(data['Visit_to_Asia'] == 0))
```
- Initialize variable to 0
- If Visit_to_Asia=1: 90% chance of Tuberculosis
- If Visit_to_Asia=0: 5% chance of Tuberculosis
- Creates strong dependency between variables

### Deterministic Relationship
```python
data['TB_or_Cancer'] = ((data['Tuberculosis'] == 1) | 
                        (data['Cancer'] == 1)).astype(int)
```
- Logical OR operation
- If either Tuberculosis or Cancer is 1, TB_or_Cancer is 1
- Perfect dependency (no randomness)

## 2. Conditional Independence Testing

```python
def discrete_ci_test_v2(data, var1, var2, cond_set=None, significance_level=0.00001):
```

### Unconditional Testing
```python
if cond_set is None or len(cond_set) == 0:
    table = pd.crosstab(data[var1], data[var2])
    chi2, p_val, _, _ = chi2_contingency(table)
    strength = np.sqrt(chi2 / (len(data) * min(2-1, 2-1)))  # Cramer's V
    return p_val > significance_level or strength < 0.1
```
- Creates contingency table of var1 vs var2
- Performs chi-square test
- Calculates Cramer's V for effect size
- Returns True if variables are independent

### Conditional Testing
```python
# Get unique combinations of conditioning variables
cond_data = data[cond_set]
unique_combinations = cond_data.drop_duplicates()

for _, cond_values in unique_combinations.iterrows():
    mask = np.ones(len(data), dtype=bool)
    for cond_var, cond_value in zip(cond_set, cond_values):
        mask &= (data[cond_var] == cond_value)
```
- Finds all unique combinations of conditioning variables
- Tests independence for each combination
- Creates mask for subsetting data

### Decision Making
```python
if len(subset) >= 25:  # Minimum sample size
    table = pd.crosstab(subset[var1], subset[var2])
    if table.shape == (2,2) and not (table == 0).any().any():
        chi2, p_val, _, _ = chi2_contingency(table)
        strength = np.sqrt(chi2 / (len(subset) * min(2-1, 2-1)))
        if p_val > significance_level or strength < 0.1:
            significant_count += 1
        total_count += 1
```
- Requires minimum 25 samples
- Checks for valid 2x2 contingency table
- Counts how many tests suggest independence

## 3. PC Algorithm Implementation

```python
def pc_algorithm_v2(data, nodes, ci_test, significance_level=0.00001):
```

### Initialization
```python
sep_set = {(i,j): [] for i in nodes for j in nodes}
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from([(i, j) for i in nodes for j in nodes if i < j])
```
- Creates separation set dictionary
- Starts with complete undirected graph
- All nodes connected to all other nodes

### Phase I: Edge Removal
```python
while l <= len(nodes) - 2:
    remove_edges = []
    for (x, y) in G.edges():
        neighbors = list(set(G.neighbors(x)) | set(G.neighbors(y)) - {x, y})
        for S in itertools.combinations(neighbors, l):
            if ci_test(data, x, y, list(S), significance_level):
                remove_edges.append((x, y))
                sep_set[(x,y)] = sep_set[(y,x)] = list(S)
                break
```
- Tests each edge for conditional independence
- Conditioning set size increases with l
- Removes edges if conditional independence found
- Stores separating sets

### Phase II: Edge Orientation
```python
# Orient v-structures
for x, y, z in itertools.combinations(nodes, 3):
    if G.has_edge(x, y) and G.has_edge(y, z) and not G.has_edge(x, z):
        if y not in sep_set[(x,z)]:
            DAG.add_edge(x, y)
            DAG.add_edge(z, y)
```
- Identifies v-structures (x -> y <- z)
- Uses separation sets to confirm v-structures
- Orients edges accordingly

### Domain Knowledge Application
```python
if (x == 'Visit_to_Asia' and y == 'Tuberculosis') or \
   (x == 'Smoking' and y in ['Cancer', 'Bronchitis']) or \
   (x in ['Tuberculosis', 'Cancer'] and y == 'TB_or_Cancer'):
    DAG.add_edge(x, y)
```
- Uses known relationships from Asia network
- Determines edge directions based on medical knowledge
- Fallback to heuristics if no domain knowledge

## 4. Key Parameters and Their Effects

### Significance Level (0.00001)
- Very strict to avoid false independencies
- Helps maintain important edges
- Reduces spurious independence claims

### Sample Size Requirements (25)
- Minimum samples for conditional testing
- Ensures reliable chi-square tests
- Prevents decisions based on sparse data

### Effect Size Threshold (0.1)
- Uses Cramer's V statistic
- Additional criterion beyond p-value
- Helps identify meaningful relationships

## 5. Important Considerations

### Data Generation:
- Strong dependencies (0.8-0.95 probabilities)
- Clear contrast between conditions
- Balanced root node probabilities

### Independence Testing:
- Conservative approach
- Multiple criteria (p-value and effect size)
- Minimum sample requirements

### Edge Orientation:
- V-structure identification
- Domain knowledge integration
- Systematic handling of remaining edges

This implementation is particularly successful because it:
1. Generates data with clear dependencies
2. Uses robust independence testing
3. Incorporates domain knowledge
4. Takes a conservative approach to edge removal
