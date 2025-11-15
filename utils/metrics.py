import math


def psnr(mse):
    """
    Compute PSNR (in dB) from MSE.
    """
    return -10.0 * math.log10(max(mse, 1e-12))