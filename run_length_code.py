import numpy as np


def run_length_encode(symb_index: np.ndarray, K) -> np.ndarray:
    """
    Encodes an input numpy array using a simple run-length encoding algorithm,
    specifically designed for encoding sequences of zeros. The function
    returns a numpy array of size n x 2, where n is the number of runs of zeros
    in the input array, and each row represents a run of zeros, with the first
    column indicating the starting index of the run and the second column
    indicating the length of the run.

    Args:
        symb_index (np.ndarray): A numpy array to be encoded using run-length
            encoding.
        K (int): The length of the input numpy array.

    Returns:
        An n x 2 numpy array, where n is the number of runs of zeros in the input
        array, and each row represents a run of zeros, with the first column
        indicating the starting index of the run and the second column indicating
        the length of the run.
    Examples:
        >>> arr = np.array([0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 4])
        >>> K = arr.shape[0]
        >>> run_length_encode(arr, K)
        array([[0, 0],
               [1, 6],
               [2, 1],
               [1, 4],
               [4, 0]])

    """
    # Initialize an empty list to hold the run-length encoded symbols
    rle = []

    # Initialize the loop counter to 0
    i = 0

    # Iterate over the entire input array
    while i < K:

        # Initialize the length of the current run of zeros
        current_run_length = 1

        # Keep incrementing the length as long as the next element is zero
        while i + current_run_length < K and symb_index[i + current_run_length] == 0:
            current_run_length += 1

        # Add the current symbol and its run length to the run-length encoded symbols list
        rle.append([symb_index[i], current_run_length - 1])

        # Update the index to start after the current run of zeros
        i += current_run_length

    return np.array(rle).astype(int)


def run_length_decode(run_lengths: np.ndarray, K) -> np.ndarray:
    """
    Decodes a run-length encoded array of zeros.

    Args:
        run_lengths (np.ndarray): A numpy array of run lengths.
        K (int): The length of the decoded sequence.

    Returns:
        A numpy array containing the decoded sequence.

    Examples:
        >>> arr = np.array([[1, 0], [1, 1], [3, 1], [4, 0], [1, 5], [1, 4], [2, 2]])
        >>> K = arr.shape[0]
        >>> run_length_decode(arr, K)
        array([1, 1, 0, 3, 0, 4, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0])

    """
    # Initialize an empty array to store the decoded sequence.
    decoded_sequence = np.array([])

    # Loop through each run-length pair in the encoded sequence.
    for i in range(K):
        # Append the non-zero symbol to the decoded sequence.
        decoded_sequence = np.append(decoded_sequence, run_lengths[i, 0])

        # Append the appropriate number of zeros to the decoded sequence.
        decoded_sequence = np.append(decoded_sequence, run_lengths[i, 1] * [0])

    return decoded_sequence.astype(int)
