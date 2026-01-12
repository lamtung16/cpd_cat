import numpy as np
from scipy.linalg import sqrtm, inv, logm, expm

# ------------------------------ DISTANCE AND MEAN ------------------------------
# distance between two data points
def distance(a, b, distance_kind):
    match distance_kind:
        case "euclid":
            return np.sqrt(np.sum((a - b) ** 2))
        
        case "circular":
            diff = np.abs(a - b)
            return np.sum(np.minimum(diff, 2 * np.pi - diff) ** 2)
        
        case "sphere":
            dot = np.dot(a/np.linalg.norm(a), b/np.linalg.norm(b))
            dot = np.clip(dot, -1.0, 1.0)
            return np.arccos(dot) ** 2
        
        case "grassmann":
            S = np.linalg.svd(a.T @ b, compute_uv=False)
            return np.linalg.norm(np.arccos(np.clip(S, -1.0, 1.0)))

        case "spd_affine":
            wA, QA = np.linalg.eigh(a)
            A_inv_sqrt = QA @ np.diag(1.0 / np.sqrt(wA)) @ QA.T
            C = A_inv_sqrt @ b @ A_inv_sqrt
            wC = np.linalg.eigvalsh(C)
            return np.linalg.norm(np.log(wC))


# mean of a segment
def mean(seg, distance_kind):
    max_iter = 100
    tol = 10e-7
    
    match distance_kind:
        case "euclid":
            return np.mean(seg, axis=0)
        
        case "circular":
            mean_angles = np.zeros(seg.shape[1])
            for j in range(seg.shape[1]):
                s, c = np.mean(np.sin(seg[:, j])), np.mean(np.cos(seg[:, j]))
                ang = np.arctan2(s, c)
                if ang < 0:
                    ang += 2 * np.pi
                mean_angles[j] = ang
            return mean_angles
        
        case "sphere":
            mu = np.mean(seg, axis=0)
            mu /= np.linalg.norm(mu)
            for _ in range(max_iter):
                grad = np.zeros_like(mu)
                for x in seg:
                    dot = np.clip(np.dot(mu, x), -1.0, 1.0)
                    theta = np.arccos(dot)
                    if theta > tol:
                        grad += theta / np.sin(theta) * (x - dot * mu)
                grad /= len(seg)
                if np.linalg.norm(grad) < tol:
                    break
                normg = np.linalg.norm(grad)
                mu = np.cos(normg) * mu + np.sin(normg) * (grad / normg)
                mu /= np.linalg.norm(mu)
            return mu
        
        case "grassmann":
            M = seg[0]
            for _ in range(max_iter):
                tangent_sum = np.zeros_like(M)
                for X in seg:
                    U, S, Vt = np.linalg.svd(M.T @ X)
                    tangent = X - M @ (U @ Vt)
                    tangent_sum += tangent
                tangent_norm = np.linalg.norm(tangent_sum)
                if tangent_norm < tol:
                    break
                M += tangent_sum / len(seg)
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                M = U @ Vt
            return M

        case "spd_affine":
            M = seg[0]
            for _ in range(max_iter):
                sum_log = np.zeros_like(M)
                sqrt_M = sqrtm(M)
                inv_sqrt_M = inv(sqrt_M)
                for X in seg:
                    log_arg = inv_sqrt_M @ X @ inv_sqrt_M
                    sum_log += np.real(logm(log_arg))
                sum_log /= len(seg)
                M_new = sqrt_M @ expm(sum_log) @ sqrt_M
                M_new = np.real(M_new)
                if np.linalg.norm(M_new - M) < tol:
                    break
                M = M_new
            return M

# ------------------------------ CAT-OP ------------------------------
# centroids: mean of each signal segment
def init_mean(signal, distance_kind, n_states):
    T = len(signal)
    centroids = [None] * n_states
    seg_size = T // n_states
    for i in range(n_states):
        start = i * seg_size
        end = T if i == n_states - 1 else (i + 1) * seg_size
        seg = signal[start:end]
        centroids[i] = mean(np.array(seg), distance_kind)
    return centroids


# main changepoint detection algorithm
def cpd_cat(signal, distance_kind, n_states, pen):
    centroids = init_mean(signal, distance_kind, n_states)

    T = len(signal)
    M = len(centroids)

    V = np.zeros((T + 1, M))
    tau = -1 * np.ones((T + 1, M), dtype=np.int32)
    last_change = -1 * np.ones((T), dtype=np.int32)

    best_prev = 0
    for t in range(1, T + 1):
        for k in range(M):
            if best_prev + pen < V[t - 1][k]:
                V[t][k]   = best_prev + pen
                tau[t][k] = t - 2
            else:
                V[t][k]   = V[t - 1][k]
                tau[t][k] = tau[t - 1][k]
            
            V[t][k] = V[t][k] + distance(centroids[k], signal[t-1], distance_kind)

        best_idx = np.argmin(V[t])
        best_prev = V[t][best_idx]
        last_change[t-1] = tau[t][best_idx]
    
    # trace back
    s = last_change[-1]
    chpnts = np.array([], dtype=np.int32)
    while s > 0:
        chpnts = np.append(s, chpnts)
        s = last_change[s]

    return chpnts + 1


#------------------------------ DUBEY-MULLER ------------------------------
def dm_stat_exact(data: list[np.ndarray], min_size: int = 10, distance_kind: str = 'euclid'):
    """Compute renormalized T_n(u) for all candidate splits and return the best k."""
    n = len(data)

    # ---- Global Fréchet mean and variance ----
    mu_global = mean(data, distance_kind)
    d2 = np.array([distance(y, mu_global, distance_kind) for y in data])
    V_global = d2.mean()
    # Renormalization factor: variance of squared distances
    d4 = d2**2
    sigma_hat2 = d4.mean() - V_global**2

    best_stat = 0.0
    best_k = -1

    for k in range(min_size, n - min_size):
        u = k / n
        left = data[:k]
        right = data[k:]
        mu_left = mean(left, distance_kind)
        mu_right = mean(right, distance_kind)
        V_left = np.mean([distance(y, mu_right, distance_kind) for y in left])
        V_right = np.mean([distance(y, mu_left, distance_kind) for y in right])
        VC_left = np.mean([distance(y, mu_left, distance_kind) for y in left])
        VC_right = np.mean([distance(y, mu_right, distance_kind) for y in right])
        raw_stat = u * (1 - u) * (
            (V_left - V_right)**2
            + (VC_left - V_left + VC_right - V_right)**2
        )
        stat = raw_stat / sigma_hat2
        if stat > best_stat:
            best_stat = stat
            best_k = k

    return best_k, best_stat


def cpd_dm(signal: list[np.ndarray], min_size: int = 20, threshold: float = 0.05, max_changes: int = 50, distance_kind: str = 'euclid'):
    n = len(signal)
    detected = []
    segments = [(0, n)]
    while segments and len(detected) < max_changes:
        start, end = segments.pop()
        if end - start < 2 * min_size:
            continue
        segment_data = signal[start:end]
        best_k_rel, best_stat = dm_stat_exact(segment_data, min_size, distance_kind)
        if best_k_rel == -1 or best_stat < threshold:
            continue
        best_k = start + best_k_rel
        detected.append(best_k)
        segments.append((start, best_k))
        segments.append((best_k, end))
    return sorted(detected)