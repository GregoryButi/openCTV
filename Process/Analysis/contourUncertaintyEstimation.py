import numpy as np

def variance(array):
    """
    Calculate the variance of predictions or probabilities as a consensus measure.

    Args:
        array (numpy.ndarray): array of shape (S, X, Y, Z, M)

    Returns:
        numpy.ndarray: Variance of predictions of shape (S, X, Y, Z)
    """
    variance_map = np.var(array, axis=-1)  # Shape: (S, X, Y, Z)
    return variance_map

def disagreement_score(predictions, chosen_prediction):
    """
    Calculate the disagreement score as the number of models that disagree with the chosen label.

    Args:
        predictions (numpy.ndarray): Prediction array of shape (X, Y, Z, M)
        chosen_prediction (numpy.ndarray): The prediction to compare against of shape (X, Y, Z)

    Returns:
        numpy.ndarray: Disagreement score of shape (S, X, Y, Z)
    """

    disagreement_count = np.zeros(predictions.shape[0:3])
    for m in range(predictions.shape[-1]):
        disagreement_count += np.logical_and(predictions[..., m].astype(bool), ~chosen_prediction.astype(bool))

    disagreement_score = disagreement_count / predictions.shape[-1]
    return disagreement_score

def entropy(array):
    """
    Calculate the entropy of predictions or probabilities.

    Args:
        array (numpy.ndarray): array of shape (S, X, Y, Z, C)

    Returns:
        numpy.ndarray: Entropy of shape (S, X, Y, Z)
    """
    entropy_map = -np.sum(array * np.log(np.clip(array, 1e-8, 1.0)), axis=-1)
    return entropy_map

def mutual_information(predictions):
    """
    Calculate mutual information from softmax outputs.

    Args:
        predictions (numpy.ndarray): Prediction array of shape (S, X, Y, Z, M, C)

    Returns:
        numpy.ndarray: Mutual information of shape (S, X, Y, Z)
    """
    mean_prob = np.mean(predictions, axis=-2)  # Average over models
    entropy_mean = entropy(mean_prob)  # Entropy of the mean prediction
    mean_entropy = np.mean(entropy(predictions), axis=-2)  # Mean entropy of individual models
    mutual_info = entropy_mean - mean_entropy
    return mutual_info