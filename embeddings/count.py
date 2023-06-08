import numpy as np

def get_total_vector_count(embeddings):
    # Method 1: Using numpy.size() function
    total_count_method_1 = np.size(embeddings)

    # Method 2: Computing the product of shape dimensions
    total_count_method_2 = np.prod(embeddings.shape)

    return total_count_method_1, total_count_method_2

# Sample usage
embeddings_filepath = "embeddings.npy"
embeddings = np.load(embeddings_filepath)
total_count_1, total_count_2 = get_total_vector_count(embeddings)
print("Total vector count (Method 1):", total_count_1)
print("Total vector count (Method 2):", total_count_2)