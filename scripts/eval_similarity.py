import numpy as np


def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors.

    Args:
      a: The first vector.
      b: The second vector.

    Returns:
      The cosine similarity between a and b.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def euclidean_distance(a, b):
    """Calculates the Euclidean distance between two vectors.

    Args:
      a: The first vector (numpy array).
      b: The second vector (numpy array).

    Returns:
      The Euclidean distance between a and b (float).
    """
    return np.linalg.norm(a - b)


def lerp(start, end, t):
    """
    Performs linear interpolation between start and end.

    Returns:
      The interpolated value (scalar or NumPy array, depending on inputs).
    """
    return start + (end - start) * t


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        t (float/np.ndarray): Float value between 0.0 and 1.0
        v0 (np.ndarray): Starting vector
        v1 (np.ndarray): Final vector
        DOT_THRESHOLD (float): Threshold for considering the two vectors as
                               colineal. Not recommended to alter this.
    Returns:
        v2 (np.ndarray): Interpolation vector between v0 and v1
    '''
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(v0, v1, t)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    return v2


def normalize_2d(tensor):
    norms = np.linalg.norm(tensor, axis=1, keepdims=True)
    return tensor / norms


def slerp_average(tensor, t=0.5):
    """
    Performs slerp between all pairs and averages the results in given dimension.

    Args:
      tensor: The input tensor of shape (n, d).
      t: The interpolation parameter (between 0 and 1).

    Returns:
      A tensor of shape (1, d) representing the interpolated point.
    """
    num_points = tensor.shape[0]
    if num_points <= 1:
        return tensor
    interpolated_points = []
    for i in range(num_points):
        for j in range(i + 1, num_points):
            interpolated_points.append(slerp(t, tensor[i].numpy(), tensor[j].numpy()))
    result = np.mean(interpolated_points, axis=0)
    return result.reshape(1, -1)


def euclidean_average(tensor):
    """
    Calculates the simple arithmetic mean (centroid) of the vectors.
    This is equivalent to slerp_average(tensor, t=0.5) for when n is big.

    Args:
      tensor: The input tensor of shape (n, d).

    Returns:
      A tensor of shape (1, d) representing the mean point.
    """
    return tensor.mean(axis=0).reshape(1, -1)


def evaluate_pairs_b(pairs, use_appr=False, f_out=None):
    assert len(pairs[1][1]) == 1 and len(pairs[2][1]) == 1 and len(pairs[0][1]) == 1
    test_o = pairs[0][1][0]
    true_os = pairs[1][1][0]
    false_os = pairs[2][1][0]

    # print(len(test_o), len(test_o[0]), test_o[0][-1].shape)  # module, layers

    if use_appr:
        agg = lambda xs: normalize_2d(euclidean_average(xs.to("cpu").float().detach()))[0]
    else:
        agg = lambda xs: slerp_average(normalize_2d(xs.to("cpu").float().detach()), 0.5)[0]

    in_shape = test_o.shape
    test_vector = agg(test_o)
    out_shape = test_vector.shape
    true_vector = agg(true_os)
    false_vector = agg(false_os)

    # min_true = euclidean_distance(test_vector, true_vector)
    # min_false = euclidean_distance(test_vector, false_vector)
    # print(f"in: {in_shape}, out: {out_shape}, min_true: {min_true}, min_false: {min_false}, correct: {min_true < min_false}")
    # return min_true < min_false

    max_true = cosine_similarity(test_vector, true_vector)
    max_false = cosine_similarity(test_vector, false_vector)
    if f_out:
        f_out.write(f"in: {in_shape}, out: {out_shape}, max_true: {max_true}, max_false: {max_false}, correct: {max_true > max_false}\n")
    else:
        print(f"in: {in_shape}, out: {out_shape}, max_true: {max_true}, max_false: {max_false}, correct: {max_true > max_false}")
    return max_true > max_false


def evaluate_pairs_s(pairs, use_appr=False, f_out=None):
    test_o = pairs[0][1][0]
    true_os = pairs[1][1]
    false_os = pairs[2][1]

    # print(len(test_o), len(test_o[0]), test_o[0][-1].shape)  # module, layers

    if use_appr:
        agg = lambda xs: normalize_2d(euclidean_average(xs.to("cpu").float().detach()))[0]
    else:
        agg = lambda xs: slerp_average(normalize_2d(xs.to("cpu").float().detach()), 0.5)[0]

    in_shape = test_o.shape
    test_vector = agg(test_o)
    out_shape = test_vector.shape
    true_vectors = [agg(true_os[x]) for x in range(len(true_os))]
    false_vectors = [agg(false_os[x]) for x in range(len(false_os))]

    max_true = max([cosine_similarity(test_vector, vector) for vector in true_vectors])
    max_false = max([cosine_similarity(test_vector, vector) for vector in false_vectors])
    if f_out:
        f_out.write(f"in: {in_shape}, out: {out_shape}, len: {len(true_vectors), len(false_vectors)}, max_true: {max_true}, max_false: {max_false}, correct: {max_true > max_false}\n")
    else:
        print(f"in: {in_shape}, out: {out_shape}, len: {len(true_vectors), len(false_vectors)}, max_true: {max_true}, max_false: {max_false}, correct: {max_true > max_false}")
    return max_true > max_false
