import torch

# Old version of batched_kronecker (fixed)
def batched_kronecker_old(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    
    # Batch matrix multiplication (bmm)
    kronecker_prod = torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1))

    # Correct the reshaping part
    batch_size = matrix1.size(0)
    n1 = matrix1.size(1)
    p1 = matrix1.size(2)
    n2 = matrix2.size(1)
    p2 = matrix2.size(2)

    # Reshaping the output properly
    output = kronecker_prod.reshape(batch_size, n1 * n2, p1 * p2)

    return output

# New version of batched_kronecker (same logic as old but rechecked)
def batched_kronecker_new(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    
    # Batch matrix multiplication (bmm)
    kronecker_prod = torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1))

    # Correct the reshaping part
    batch_size = matrix1.size(0)
    n1 = matrix1.size(1)
    p1 = matrix1.size(2)
    n2 = matrix2.size(1)
    p2 = matrix2.size(2)

    # Reshaping the output properly
    output = kronecker_prod.reshape(batch_size, n1 * n2, p1 * p2)

    return output

# Create random tensors to simulate the matrices
batch_size = 2
n1, m1, p1 = 3, 3, 2  # Shape of matrix1
n2, m2, p2 = 3, 3, 2  # Shape of matrix2

matrix1 = torch.randn(batch_size, n1, m1, p1)
matrix2 = torch.randn(batch_size, n2, m2, p2)

# Test both functions
# output_old = batched_kronecker_old(matrix1, matrix2)
output_new = batched_kronecker_new(matrix1, matrix2)

# # Print results to compare
# print("Old function output shape:", output_old.shape)
print("New function output shape:", output_new.shape)

# Check if outputs are approximately equal
# print("Are the outputs equal?", torch.allclose(output_old, output_new))