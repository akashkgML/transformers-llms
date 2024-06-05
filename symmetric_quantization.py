#!/usr/bin/python3

"""
Symmetric Quantization - Restricted Mode
-----------------------------------------
This script allows user to perform symmetric quantization.
This script is self explainable and contains essential
functions that are required in the process of symmetric
quantization in restricted mode.
"""
import numpy as np
from typing import Tuple

np.random.seed(58)


def get_random_weights(size=20, low=-140, high=140):
    # type: (int, int, int) -> np.ndarray
    """
    Function to generate weights from a uniform distribution.
    This function deliberately include 0 as a weight to get a
    clear sense of the Zero-point concept.

    :param size: size of the array containing weights
    :param low: lower bound for weights
    :param high: upper bound for weights
    :return: Numpy array containing weights from uniform distribution
    """
    # Generate uniformly distributed random weights
    weights = np.random.uniform(low=low, high=high, size=size)

    # we will deliberately keep middle weight as zero for
    # the illustration purpose of Zero-point
    weights[size//2] = 0

    return weights


def clip(weights_q, lower_bound, upper_bound):
    # type: (np.ndarray, int, int) -> np.ndarray
    """
    Uniform quantization transforms the input value x ∈ [α, β]
    to lie within [q_min, q_max], where inputs outside the range
    are clipped to the nearest bound, using the clip function.

    :param weights_q: original weights to be quantized
    :param lower_bound: q_min i.e. minimum value of quantized integer
                of output range
    :param upper_bound: q_max i.e. maximum value of quantized integer
                of output range
    :return: clipped quantized array
    """
    weights_q[weights_q < lower_bound] = lower_bound
    weights_q[weights_q > upper_bound] = upper_bound

    return weights_q


def symmetric_quantization(weights, bits):
    # type: (np.ndarray, int) -> Tuple[np.ndarray, float]
    """
    Function that does the actual quantization. It takes
    an array of floating-points and convert them into
    bit-width integer values.

    :param weights: weights to be quantized
    :param bits: bit-width of the signed integer
    :return: quantized weights
    """
    # Compute the beta as maximum value in weights
    beta = np.max(np.abs(weights))

    # Compute the alpha which is simply -ve of weights
    alpha = -beta

    # Compute the minimum value of b-bit signed integer (in restricted range)
    q_max = 2 ** (bits - 1) - 1
    q_min = -q_max

    # Compute the scaling factor
    scale = (beta - alpha) / (q_max - q_min)

    # Clip the weights outside [lower bound, upper bound]
    quantized_weights = clip(np.round(weights / scale), q_min, q_max).astype(np.int32)

    return quantized_weights, scale


def symmetric_dequantize(quantized_weights, scale):
    # type: (np.ndarray, float) -> np.ndarray
    """
    Function to dequantize or reconstruct the original
    floating point numbers from quantized weights.

    :param quantized_weights: array of quantized (integer) weights
    :param scale: Scaling factor to be used
    :return: dequantize or reconstructed weights
    """
    return quantized_weights * scale


def quantization_error(weights, dequantized_weights):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    """
    Function to compute quantization error which is the
    difference between original weights and the dequantized
    or reconstructed weights.

    :param weights: original floating point weights
    :param dequantized_weights: quantized (integer valued) weights
    :return: quantization error
    """
    q_error = weights - dequantized_weights

    return q_error


def mse_quantization_error(weights, dequantized_weights):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Function to compute mean square quantization error.

    :param weights: original floating point weights
    :param dequantized_weights: reconstructed weights,
            from quantized weights
    :return: mean square error of quantization error matrix
            or array
    """
    # calculate the mean square error (mse)
    mse = np.mean((weights - dequantized_weights)**2)

    return mse


if __name__ == "__main__":

    # generate the random weights
    original_weights = get_random_weights(size=20)

    # compute the quantized weight and the corresponding scale
    quant_weights, scaling_factor = symmetric_quantization(weights=original_weights, bits=8)

    # reconstruct or dequantize the quantized weights
    dequant_weights = symmetric_dequantize(quant_weights, scaling_factor)

    # compute the quantization error
    quant_error = quantization_error(original_weights, dequant_weights)

    # compute the mean square error of quantization
    quant_mse = mse_quantization_error(original_weights, dequant_weights)

    print("---Result Summary for Symmetric Quantization - Restricted Mode---")

    print(f"original weights:\n {np.round(original_weights,2)}")

    print(f'symmetric scale: {scaling_factor}')

    print(f"quantized weights:\n {quant_weights}")

    print(f'de-quantized weights:\n {np.round(dequant_weights,2)}')

    print(f'quantization error:\n {np.round(quant_error, 2)}')

    print(f'quantization mean square error: {np.round(quant_mse, 2)}')
