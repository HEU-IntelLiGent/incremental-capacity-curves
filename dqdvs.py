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




def bin_array(x:np.ndarray, bin_size:float)->np.ndarray:
    """
    Bin the input array x to specified intervals.

    Rounds values to the nearest multiple of bin_size and ensures 
    the output is either monotonically increasing or decreasing 
    based on the increasing flag.

    Parameters:
    -----------
    x : np.ndarray
        Input array to be binned.
    
    bin_size : float
        Size of the bins.

    increasing : bool
        If True, output is monotonically increasing; if False, decreasing.

    Returns:
    --------
    np.ndarray
        Binned array.
    """

    x_bin = (x / bin_size).round() * bin_size #round to intervals of size bin_size
    increasing = bool(x[-1] > x[0])

    if increasing:
        return np.maximum.accumulate(x_bin) #ensure array is monotonically increasing
    else:
        return np.minimum.accumulate(x_bin)




def remove_duplicate_voltages(q: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove duplicate voltage values from the arrays q and v.

    Flags adjacent duplicate voltage entries in the monotonic voltage array 
    and retains only unique values.

    Parameters:
    -----------
    q : np.ndarray
        Cumulative capacity data corresponding to the voltages.
    
    v : np.ndarray
        Monotonic voltage data.

    Returns:
    --------
    tuple[np.ndarray, np.ndarray]
        Two arrays with duplicates removed: (q, v).
    """
    mask = np.concatenate((np.diff(v) != 0, [True]))  # flags duplicate voltage values
    return q[mask], v[mask]



def compute_dqdv_binning(q:np.ndarray, v:np.ndarray, v_increases:bool)->np.ndarray:
    """
    Compute the rate of change dq/dV based on cumulative capacity and voltage.

    Bins the voltage values, removes duplicates, and calculates the 
    numerical first differences of cumulative capacity relative to 
    voltage.

    Parameters:
    -----------
    q : np.ndarray
        Cumulative capacity data.
    
    v : np.ndarray
        Voltage data, which can be either increasing or decreasing.
    
    v_increases : bool
        Indicates if the voltage values are increasing.

    Returns:
    --------
    np.ndarray
        Array of dq/dV values after processing.
    """

    bin_size = optimal_bin_size(v)
    v_bin = bin_array(v, bin_size, v_increases)
    q_nodup, v_nodup = remove_duplicate_voltages(q, v_bin)
    return dydx_simple_differences(x=v_nodup, y=q_nodup)



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


