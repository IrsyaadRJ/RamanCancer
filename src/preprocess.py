import numpy as np
from airPLS import airPLS
import scipy as scipy
import scipy.stats as stats


# Set the seed value
seed_value = 7
np.random.seed(seed_value)

def preprocess(raw_data_frame):
    """
    Preprocess the data according to Sara: median filter for cosmic rays, airPLS for background subtraction, zscore to standardise.
    """
    datanp = np.array(raw_data_frame.iloc[4:],dtype=float)
    datamedfilt = scipy.ndimage.median_filter(datanp,size=(1,5))
    baseline = np.zeros_like(datamedfilt)
    cols = baseline.shape[1]
    for col in range(cols):
        baseline[:,col] = airPLS(datamedfilt[:,col], lambda_=300)
    data_bksb = datamedfilt-baseline
    normed = stats.zscore(data_bksb, axis=0)

    return normed


