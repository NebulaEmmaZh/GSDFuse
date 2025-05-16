import numpy as np
from imblearn.over_sampling import SMOTE
from utils import printf # Assuming printf is available globally or in a common utils

def apply_smote(features_np, labels_1d_np, k_neighbors=5, random_state=None):
    """
    Apply SMOTE to generate synthetic samples.

    Args:
        features_np (np.ndarray): NumPy array of original features (num_samples, feature_dim).
        labels_1d_np (np.ndarray): NumPy array of 1D labels (num_samples,).
        k_neighbors (int): Number of nearest neighbors for SMOTE.
                           This will be adjusted if any class has fewer samples than k_neighbors + 1.
        random_state (int, optional): Random state for reproducibility.

    Returns:
        tuple: (synthetic_features_np, synthetic_labels_1d_np)
               - synthetic_features_np (np.ndarray): Features of the generated synthetic samples.
                                                     Returns original features if SMOTE cannot be applied or generates no new samples.
               - synthetic_labels_1d_np (np.ndarray): Labels of the generated synthetic samples.
                                                     Returns original labels if SMOTE cannot be applied.
    """
    printf(f"Attempting SMOTE: Input features shape: {features_np.shape}, Input labels shape: {labels_1d_np.shape}", style="blue")

    if features_np.shape[0] == 0 or labels_1d_np.shape[0] == 0:
        printf("  SMOTE Warning: Empty features or labels array. Returning empty arrays.", style="yellow")
        return np.array([]).reshape(0, features_np.shape[1] if features_np.ndim > 1 else 0), np.array([])

    unique_labels, counts = np.unique(labels_1d_np, return_counts=True)

    if len(unique_labels) <= 1:
        printf("  SMOTE Warning: SMOTE requires more than 1 class to operate. Returning original data.", style="yellow")
        # SMOTE cannot run with a single class. It might be better to return original features
        # or handle this case based on specific needs. For now, return empty, as if no *new* samples generated.
        return np.array([]).reshape(0, features_np.shape[1]), np.array([])


    min_class_count = counts.min()
    
    # Adjust k_neighbors: SMOTE's k_neighbors must be less than the number of samples in the smallest class.
    # k_neighbors defaults to 5. If min_class_count is 6, k_neighbors can be up to 5.
    # If min_class_count is 5, k_neighbors must be < 5 (e.g., 4).
    # If min_class_count is 1, SMOTE cannot run with k_neighbors > 0.
    adjusted_k_neighbors = k_neighbors
    if min_class_count <= k_neighbors : # if samples are 5, k_neighbors is 5, then k must be < 5.
        adjusted_k_neighbors = max(1, min_class_count - 1) # Must be at least 1 if min_class_count > 1
        printf(f"  SMOTE Info: Adjusting k_neighbors from {k_neighbors} to {adjusted_k_neighbors} due to smallest class size ({min_class_count}).", style="yellow")

    if adjusted_k_neighbors == 0: # This happens if min_class_count was 1
        printf("  SMOTE Warning: Cannot apply SMOTE as adjusted k_neighbors is 0 (smallest class may have only 1 sample). Returning original data.", style="yellow")
        return np.array([]).reshape(0, features_np.shape[1]), np.array([])

    try:
        # Ensure sampling_strategy is not attempting to oversample classes that don't exist or are too small for k_neighbors
        # A common strategy is 'auto' or 'not majority' or a dict specifying counts for minority classes.
        # 'auto' is equivalent to 'not majority'.
        smote_sampler = SMOTE(k_neighbors=adjusted_k_neighbors, random_state=random_state, sampling_strategy='auto')
        
        # fit_resample will oversample all classes that are not the majority class.
        # The resampled data will contain original majority samples and oversampled minority samples.
        resampled_features_np, resampled_labels_1d_np = smote_sampler.fit_resample(features_np, labels_1d_np)
        
        printf(f"  SMOTE applied. Original feature count: {features_np.shape[0]}. Resampled feature count: {resampled_features_np.shape[0]}.") #, style="green")

        # We are interested in the *newly generated* synthetic samples.
        # imblearn's SMOTE.fit_resample returns the original samples + synthetic ones.
        # A simple way to get *only* synthetic samples is to identify which ones are new.
        # However, for simplicity in integrating with the current plan of having a "synthetic_feature_pool",
        # we might want to return *all* resampled minority class instances (original + synthetic)
        # or *only* the purely synthetic ones.
        # The prompt implies a "synthetic feature pool". If this pool is *in addition* to original data,
        # then we should try to isolate purely synthetic samples.
        # If the goal is just to have a balanced set for the "synthetic" path of the model, then `resampled_features_np` is fine.

        # For now, let's assume the goal is to get a pool of features that *includes* the synthetic additions,
        # which is what fit_resample provides (oversampled minorities + original majority).
        # Or, if the pool is meant to be *only* synthetic, then:
        num_original_samples = features_np.shape[0]
        synthetic_features_only_np = resampled_features_np[num_original_samples:]
        synthetic_labels_only_np = resampled_labels_1d_np[num_original_samples:]

        if synthetic_features_only_np.shape[0] > 0:
             printf(f"  Extracted {synthetic_features_only_np.shape[0]} purely synthetic samples.")#, style="green")
             return synthetic_features_only_np, synthetic_labels_only_np
        else:
             printf("  SMOTE did not generate any new distinct samples (possibly due to class distribution or parameters). Returning empty synthetic pool.", style="yellow")
             return np.array([]).reshape(0, features_np.shape[1]), np.array([])


    except ValueError as e:
        printf(f"  SMOTE Error: {e}. Returning empty synthetic pool.", style="red")
        return np.array([]).reshape(0, features_np.shape[1]), np.array([]) 