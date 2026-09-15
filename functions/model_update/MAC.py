import numpy as np

def MAC(phi1, phi2):
    """
    Modal Assurance Criterion between two mode shape vectors.
 
    Returns a scalar in [0,1]; 1 = perfectly correlated (same shape,
    up to arbitrary scale/sign), 0 = orthogonal/unrelated.
 
    Insensitive to scaling and sign (MAC(phi, -phi) = MAC(phi, phi) = 1),

    """
    phi1 = np.asarray(phi1).flatten()
    phi2 = np.asarray(phi2).flatten()
 
    if phi1.shape != phi2.shape:
        raise ValueError(f"phi1 and phi2 must have the same length, got {phi1.shape} and {phi2.shape}")
 
    # Hermitian inner products (np.vdot conjugates its first argument)
    cross = np.vdot(phi1, phi2)
    norm1 = np.vdot(phi1, phi1).real
    norm2 = np.vdot(phi2, phi2).real
 
    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("MAC is undefined for a zero mode shape vector")
 
    return np.abs(cross) ** 2 / (norm1 * norm2)