# Changepoint Detection Algorithm (CAT)

## Algorithm Description

### Inputs:

- `signal`: array of numeric values (time series)  
- `distance_kind`: Type of distance metric to use  
- `n_states`: Number of states (centroids) to initialize  
- `pen`: Penalty value to penalize changepoint  

### Output:

- `chpnts`: Array of detected changepoints

---

### Pseudo-code

```text
FUNCTION cpd(signal, distance_kind, n_states, pen)
    # Initialize centroids for n_states
    centroids ← init_mean(signal, distance_kind, n_states)

    T ← length of signal
    M ← number of centroids

    # Initialize DP tables
    V ← zeros((T + 1, M))        # cumulative cost table
    tau ← -1 * ones((T + 1, M))  # last change index table
    last_change ← -1 * ones(T)   # record last changepoint for each t

    best_prev ← 0

    # Dynamic programming loop
    FOR t FROM 1 TO T
        FOR k FROM 0 TO M-1
            IF best_prev + pen < V[t-1][k]
                V[t][k] ← best_prev + pen
                tau[t][k] ← t - 2
            ELSE
                V[t][k] ← V[t-1][k]
                tau[t][k] ← tau[t-1][k]
            
            # Add cost of current point to centroid
            V[t][k] ← V[t][k] + distance(centroids[k], signal[t-1], distance_kind)
        
        # Find best cumulative cost at time t
        best_idx ← index of min(V[t])
        best_prev ← V[t][best_idx]
        last_change[t-1] ← tau[t][best_idx]

    # Traceback to find changepoints
    s ← last_change[T-1]
    chpnts ← empty array

    WHILE s > 0
        prepend s to chpnts
        s ← last_change[s]

    RETURN chpnts
END FUNCTION
```

---

### Example Usage

```python
test_signal = np.column_stack([
    np.concatenate([np.random.normal(m, 0.3, 5) for m in [0, 2, 4]]),
    np.concatenate([np.random.normal(m, 0.3, 5) for m in [5, 0, 3]])
]) % (2 * np.pi)

chpnts = cpd(signal = test_signal,
             distance_kind = 'circular', 
             n_states = 5, 
             pen = 1.0)

print(chpnts)
```
```
[ 5 10]
```