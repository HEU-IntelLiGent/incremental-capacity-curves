import polars as pl
import numpy as np
import scipy as sp




def dqdv_finite_differences(q:np.ndarray, v:np.ndarray)->np.ndarray:
    """
    Compute the numerical first differences dq/dv.

    Pads the result with the last value to match the size of the input arrays q and v.

    Parameters:
    -----------
    q : np.ndarray
        Array in numerator.
    
    v : np.ndarray
        Array in denominator.

    Returns:
    --------
    np.ndarray, np.ndarray
        Arrays of voltages and dq/dv's.
    """
    dqdv = np.diff(q)/np.diff(v)
    dqdv = np.append(dqdv, dqdv[-1]) #repeat last value of dy/dx so arrays have the same size
    mask = np.isfinite(dqdv)

    return v[mask], dqdv[mask] 



def dqdv_central_differences(q:np.ndarray, v:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    Compute the numerical first differences dq/dv.

    Pads the result with the last value to match the size of the input arrays q and v.

    Parameters:
    -----------
    q : np.ndarray
        Array in numerator.
    
    v : np.ndarray
        Array in denominator.

    Returns:
    --------
    np.ndarray, np.ndarray
        Arrays of voltages and dq/dv's.
    """

    dqdv = np.gradient(q, v)
    mask = np.isfinite(dqdv)    
    return v[mask], dqdv[mask]




def compute_monotonic_ocv(q:np.ndarray, v:np.ndarray, epsilon:float):
    """
    Computes a monotonic OCV by 1) applying a cumulative monotonic envelope (running max for increasing series, 
    running min for decreasing) and 2) eliminating rapidly changing voltages (diff(V) < epsilon).
    
    Args:
        q (np.ndarray): Array of capacities.
        v (np.ndarray): Array of voltages.
        epsilon (float): The minimum voltage value to replace rapidly changing values.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the capacity values at monotonic points and the monotonic voltage values.
    """

    v_increasing = bool(v[-1] > v[0])

    if v_increasing:
        v_mon = np.maximum.accumulate(v)
        idx_keep = np.where(np.diff(v_mon) > epsilon)[0]
    else:
        v_mon = np.minimum.accumulate(v)
        idx_keep = np.where(np.diff(v_mon) < -epsilon)[0]
        
    return q[idx_keep], v_mon[idx_keep]



def dqdv_histogram(q:np.ndarray, v:np.ndarray, bin_size:float, smooth:bool=False)->tuple[np.ndarray, np.ndarray]:
    
    """
    Compute a smoothed differential capacity (dQ/dV) curve using a histogram-based method.

    This approach avoids numerical differentiation by binning voltage values and scaling
    by the total capacity range. An optional Gaussian filter is applied for smoothing.

    Args:
        q (np.ndarray): Array of capacity values (monotonic).
        v (np.ndarray): Array of voltage values corresponding to `q`.
        bin_size (float): Width of voltage bins for histogram computation.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - bin_centers (np.ndarray): Centers of voltage bins.
            - counts (np.ndarray): Smoothed dQ/dV values for each bin.
    """

    nbins = int((v.max()-v.min())/bin_size)

    counts, bin_edges = np.histogram(v, bins=nbins, density=True)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    counts *= (q.max()-q.min())

    if smooth: #optional smoothing with gaussian filter
        counts = sp.ndimage.gaussian_filter1d(counts, sigma=1) 

    v_increasing = bool(v[-1]>v[0])

    if v_increasing:
        return bin_centers, counts
    else:
        return bin_centers[::-1], -counts[::-1]


