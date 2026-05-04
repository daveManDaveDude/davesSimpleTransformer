def add_vectors(vector1: list[int], vector2: list[int]) -> list[int]:
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    return_vector = []
    for i in range(len(vector1)):
        return_vector = return_vector + [vector1[i] + vector2[i]]
    return return_vector

def dot_product(vector1: list[int], vector2: list[int]) -> int:
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    total: int = 0
    for i in range(len(vector1)):
        total = total + vector1[i] * vector2[i]
    return total


def transpose_matrix(matrix1: list[list[int]]) -> list[list[int]]:
    if not matrix1 or len(matrix1) == 0:
        raise ValueError("Matrix must be a non-empty list of lists")
    
    num_rows = len(matrix1)
    num_cols = len(matrix1[0])
    return_matrix = []
    
    for i in range(num_rows):
        if len(matrix1[i]) != num_cols:
            raise ValueError("All rows in the matrix must have the same number of columns")
    
    for i in range(num_cols):      
        return_matrix = return_matrix + [[matrix1[j][i] for j in range(num_rows)]] 
        
    return return_matrix

def matrix_vector_multiply(matrix1: list[list[int]], vector1: list[int]) -> list[int]:
    if not matrix1 or len(matrix1) == 0:
        raise ValueError("Matrix must be a non-empty list of lists")

    num_rows = len(matrix1)
    num_cols = len(matrix1[0])
    return_vector = []

    for i in range(num_rows):
        if len(matrix1[i]) != num_cols:
            raise ValueError("All rows in the matrix must have the same number of columns")

    if num_cols != len(vector1):    
        raise ValueError("Vectors must be of the same length")
    
    for i in range(num_rows):
        total = 0
        total = total + dot_product(matrix1[i], vector1)
        return_vector = return_vector + [total]
        
    return return_vector

