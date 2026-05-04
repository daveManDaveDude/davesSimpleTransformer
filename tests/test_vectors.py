import pytest

from tiny_transformer.vector import add_vectors, dot_product, matrix_vector_multiply, transpose_matrix

vector1 = [1, 2, 3]
vector2 = [10, 20, 30]
vector_different_shape = [10, 20]

vector3 = [1, 2, 3]
vector4 = [4, 5, 6]

matrix1 = [[1, 2, 3], 
           [4, 5, 6]]

matrix2 = [[1, 2],
           [3, 4], 
           [4, 5]]

matrix3 = [[1, 2, 3], 
           [4, 56 ]]

matrix4 = [[1, 2],
           [3, 4, 5], 
           [4, 5]]

matrix5 = [[1, 2, 3, 4], 
           [4, 5, 5, 6]]

matrix6 = [[1, 2],
           [3, 4], 
           [4, 5],
           [5,7 ]]

def test_add_vectors():
    result = add_vectors(vector1, vector2)
    print (result)
    assert result == [11, 22, 33]

def test_dot_product():
    result = dot_product(vector3, vector4)
    print (result)
    assert result == 32

def test_transpose_matrix_2_3():
    result = transpose_matrix(matrix1)
    print (result)
    assert result == [[1, 4], [2, 5], [3, 6]]   

def test_transpose_matrix_2_4():
    result = transpose_matrix(matrix5)
    print (result)
    assert result == [[1, 4], [2, 5], [3, 5], [4, 6]]   

def test_transpose_matrix_4_2():
    result = transpose_matrix(matrix6)
    print (result)
    assert result == [[1, 3, 4, 5], [2, 4, 5, 7]]

def test_transpose_broken_matrix_2_3():
    with pytest.raises(ValueError, match="All rows in the matrix must have the same number of columns"):
        transpose_matrix(matrix3)

def test_transpose_broken_matrix_3_2():
    with pytest.raises(ValueError, match="All rows in the matrix must have the same number of columns"):
        transpose_matrix(matrix4)

def test_transpose_matrix_3_2():
    result = transpose_matrix(matrix2) 
    print (result)
    assert result == [[1, 3, 4], [2, 4, 5]] 

def test_matrix_vector_multiply():
    result = matrix_vector_multiply([[1, 2], [3, 4]], [10, 20]) 
    assert result == [50, 110] 

def test_matrix_vector_multiply_jagged_matrix():
    with pytest.raises(ValueError, match="All rows in the matrix must have the same number of columns"):
        matrix_vector_multiply([[1, 2], [3, 4, 5]], [10, 20])

def test_matrix_vector_multiply_shape_mismatch():
    with pytest.raises(ValueError, match="Vectors must be of the same length"):
        matrix_vector_multiply([[1, 2, 2], [3, 4, 3]], [10, 20])

def test_matrix_vector_multiply_shape_mismatch2():
    with pytest.raises(ValueError, match="All rows in the matrix must have the same number of columns"):
        matrix_vector_multiply([[1, 2], [3, 4, 5]], [10, 20, 30])

def test_matrix_vector_multiply_non_square():
    result = matrix_vector_multiply([[1, 2, 3], [4, 5, 6]], [10, 20, 30]) 
    assert result == [140, 320] 

def test_add_vectors_different_shape():
    with pytest.raises(ValueError, match="Vectors must be of the same length"):
        add_vectors(vector1, vector_different_shape)    

def test_dot_product_different_shape():
    with pytest.raises(ValueError, match="Vectors must be of the same length"):
        dot_product(vector1, vector_different_shape)    

def test_transpose_matrix_wrong_input():
    with pytest.raises(ValueError, match="Matrix must be a non-empty list of lists"):
        transpose_matrix([])  

