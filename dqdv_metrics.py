import numpy as np

def absolute_percentage_error(original:np.ndarray, reconst:np.ndarray)->float:
    """    
    Compute the Absolute Percentage Error (APE) in the maximum cumulative capacity.

    Args:
        original (np.ndarray): Original cumulative capacity values.
        reconst (np.ndarray): Reconstructed cumulative capacity values.

    Returns:
        float: Absolute percentage error between the maximum values, defined as
               |max(original) - max(reconst)| / max(original).   
    
    """

    max_original = np.max(original)
    max_reconst = np.max(reconst)

    return abs(max_original - max_reconst) / max_original


def root_mean_square_error(x1:np.ndarray,
                           x2:np.ndarray,
                           y1:np.ndarray,
                           y2:np.ndarray)->float:
    
    """    
    Compute the voltage Root Mean Square Error (RMSE) between original and
    reconstructed curves over their overlapping capacity interval.

    This metric quantifies the point-wise vertical differences between the
    original curve (x1, y1) and the reconstructed curve (x2, y2). Generally
    the curves are defined on different capacity grids and ranges, the RMSE 
    is evaluated only over their common interval. The original signal is 
    first interpolated onto the reconstructed capacity grid within this overlap.

    Args:
        x1 (np.ndarray): Capacity values of the original curve (q).
        y1 (np.ndarray): Voltage values of the original curve (V).
        x2 (np.ndarray): Capacity values of the reconstructed curve (q_integrated).
        y2 (np.ndarray): Voltage values of the reconstructed curve (v).

    Returns:
        float: Root mean square error (in voltage units) computed as
               sqrt(mean((V_k - v_k)^2)), where V_k is obtained by interpolating
               (x1, y1) onto the subset of x2 within the overlapping interval.
    """
    
    x_min = max(np.min(x1), np.min(x2)) # minimum of overlapping range
    x_max = min(np.max(x1), np.max(x2)) # maximum of overlapping range

    # Keep only x2 values within the overlapping interval
    mask = (x2 >= x_min) & (x2 <= x_max)

    x2_overlap = x2[mask]
    y2_overlap = y2[mask]

    # Interpolate y1 onto the x2 grid
    y1_interp = np.interp(x2_overlap, x1, y1)

    return np.sqrt(np.mean((y1_interp - y2_overlap) ** 2))