import pytest

from tiny_transformer.vector import add_positional_embeddings, add_vectors, dot_product, embed_sequence, embed_token, matrix_multiply, matrix_vector_multiply, predict_from_logits, predict_sequence, softmax_from_logits, transpose_matrix

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

def test_matrix_multiply():
    result = matrix_multiply(
                    [[1, 2], [3, 4]],
                    [[10, 20], [30, 40]])
    assert result == [[70, 100], [150, 220]]

def test_matrix_multiply_non_square():
    result = matrix_multiply([[1, 2, 3], [4, 5, 6]],
                             [[10, 20], [30, 40], [50, 60]]
                            )
    assert result == [[220, 280], [490, 640]]

def test_matrix_multiply_empty_matrix1():
    with pytest.raises(ValueError, match="Matrices must be non-empty lists of lists"):
        matrix_multiply([], [[10, 20], [30, 40], [50, 60]] )  

def test_matrix_multiply_empty_matrix2():
    with pytest.raises(ValueError, match="Matrices must be non-empty lists of lists"):
        matrix_multiply([[10, 20], [30, 40], [50, 60]], [] )  

def test_matrix_multiply_jagged_matrix1():
    with pytest.raises(ValueError, match="All rows in the first matrix must have the same number of columns"):
        matrix_multiply(
                    [[1], [3, 4]],
                    [[10, 20], [30, 40]])  
        
def test_matrix_multiply_jagged_matrix2():
    with pytest.raises(ValueError, match="All rows in the second matrix must have the same number of columns"):
        matrix_multiply(
                    [[1, 2], [3, 4]],
                    [[10, 20], [30]])  
        
def test_matrix_multiply_shape_mismatch():
    with pytest.raises(ValueError, match="Number of columns in the first matrix must equal the number of rows in the second matrix"):
        matrix_multiply(
                    [[1, 2, 3], [3, 4, 5], [1,3, 4]],
                    [[10, 20], [10, 20 ]])  

def test_matrix_multiply_1_3_3_1():
    result = matrix_multiply([[1, 2, 3]],
                             [[10],[20],[30]])
    assert result == [[140]]

def test_predict_from_logits():
    result = predict_from_logits([1.0, 5.0, 2.0])
    assert result == 1

    result = predict_from_logits([-2.0, -1.0, -5.0])
    assert result == 1

    result = predict_from_logits([0.1, 0.2, 0.9])
    assert result == 2

def test_predict_sequence():
    result = predict_sequence([
        [1.0, 5.0, 2.0],
        [9.0, 1.0, 0.0],
    ])
    assert result == [1, 0]

def test_predict_sequence_blank_matrix():
    with pytest.raises(ValueError, match="Matrix must be a non-empty list of lists"):
        predict_sequence([])  

def test_predict_sequence_jagged_matrix():
    with pytest.raises(ValueError, match="All rows in the matrix must have the same number of columns"):
        predict_sequence([[1.0, 2.0], [3.0]]) 

def test_softmax_from_logits():
    result = softmax_from_logits([1.0, 2.0, 3.0])
    assert result == pytest.approx([
        0.09003057317038046,
        0.24472847105479767,
        0.6652409557748219,
    ])
    assert sum(result) == pytest.approx(1.0)
        
def test_logits_not_mutated_by_softmax():
    logits = [1.0, 2.0, 3.0]
    original_logits = logits.copy()
    softmax_from_logits(logits)
    assert logits == original_logits, "Logits should not be mutated by softmax_from_logits"  

def test_softmax_result_equals_input_length():
    result = softmax_from_logits([1.0, 2.0, 3.0])
    assert len(result) == 3, "Softmax result should have the same length as the input logits"     

def test_softmax_from_logits_all_probs_between_0_and_1():
    result = softmax_from_logits([1.0, 2.0, 3.0])
    assert min(result) > 0.0, "All probabilities should be greater than 0"
    assert max(result) < 1.0, "All probabilities should be less than 1"

def test_softmax_from_logits_largest_logit_has_largest_probability():  
    logits = [1.0, 2.0, 3.0]
    result = softmax_from_logits(logits)
    max_logit_index = logits.index(max(logits))
    max_prob_index = result.index(max(result))
    assert max_logit_index == max_prob_index, "The largest logit should correspond to the largest probability"

def test_embed_token():
    table = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
    result = embed_token(0, table)
    assert result == [0.1, 0.2]
    result = embed_token(2, table)
    assert result == [0.5, 0.6]

def test_embed_sequence():
    table = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
    token_ids = [2, 0, 1]
    result = embed_sequence(token_ids, table) 
    assert result == [[0.5, 0.6], [0.1, 0.2], [0.3, 0.4]]

def test_embed_token_invalid_token_id():
    table = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
    with pytest.raises(ValueError, match="Token ID must be a valid index in the embedding matrix"):
        embed_token(-1, table)
    with pytest.raises(ValueError, match="Token ID must be a valid index in the embedding matrix"):
        embed_token(3, table)

def test_embed_token_empty_table():
    table = []
    with pytest.raises(ValueError, match="Embedding matrix must be a non-empty list of lists"):
        embed_token(0, table)

def test_embed_token_jagged_table():
    table = [
        [0.1, 0.2],
        [0.3, 0.4, 0.5],
    ]
    with pytest.raises(ValueError, match="All rows in the embedding matrix must have the same number of columns"):
        embed_token(0, table)

def test_embed_sequence_empty_sequence ():
    seq = []
    table = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
    with pytest.raises(ValueError, match="Sequence must be a non-empty list of token IDs"):
        embed_sequence(seq, table)

def test_embed_token_returns_copy():
    table = [
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ]
    result = embed_token(1, table)
    result[0] = 999
    assert table[1][0] == 0.3, "Embedding matrix should not be mutated by embed_token"  
    assert table[1][1] == 0.4, "Embedding matrix should not be mutated by embed_token"  

def test_embed_sequence_non_int_token_id():
    table = [
        [0.1, 0.2, 0.4],
        [0.3, 0.4, 0.5],
    ]
    with pytest.raises(ValueError, match="Token ID must be an integer"):
        embed_sequence([0.1], table)

def test_embed_sequence_negative_token_id():
    table = [
        [0.1, 0.2, 0.4],
        [0.3, 0.4, 0.5],
    ]
    with pytest.raises(ValueError, match="Token ID must be a valid index in the embedding matrix"):
        embed_sequence([-1], table)

def test_embed_sequence_invalid_token_id():
    table = [
        [0.1, 0.2, 0.4],
        [0.3, 0.4, 0.5],
    ]
    with pytest.raises(ValueError, match="Token ID must be a valid index in the embedding matrix"):
        embed_sequence([3], table)

def test_embed_sequence_returns_copied_rows():
    table = [
        [0.1, 0.2, 0.4],
        [0.3, 0.4, 0.5],
    ]
    result = embed_sequence([1], table)
    result[0][0] = 999
    assert table[1][0] == 0.3

def test_embed_sequence_empty_matrix():
    table = []
    with pytest.raises(ValueError, match="Embedding matrix must not be empty"):
        embed_sequence([0], [])

def test_add_positional_embeddings():
    embedded_sequence = [[0.5, 0.6], [0.1, 0.2]]
    position_embedding_table = [[10.0, 10.0], [20.0, 20.0]]
    result = add_positional_embeddings(embedded_sequence, position_embedding_table)
    assert result == [[10.5, 10.6], [20.1, 20.2]]

def test_add_positional_jagged_embedded_sequence():
    embedded_sequence = [[0.5, 0.6], [0.1]]    
    position_embedding_table = [[10.0, 10.0], [20.0, 20.0]]
    with pytest.raises(ValueError, match="All rows in the embedded sequence must have the same number of columns"):
        add_positional_embeddings(embedded_sequence, position_embedding_table)

def test_add_positional_jagged_position_table():
    embedded_sequence = [[0.5, 0.6], [0.1, 0.2]]
    position_embedding_table = [[10.0, 10.0], [20.0]]
    with pytest.raises(ValueError, match="All rows in the position embedding table must have the same number of columns"):
        add_positional_embeddings(embedded_sequence, position_embedding_table)

def test_add_positional_row_count_mismatch():
    embedded_sequence = [[0.5, 0.6]]    
    position_embedding_table = [[10.0, 10.0], [20.0, 20.0]]
    with pytest.raises(ValueError, match="Embedded sequence and position embedding table must have the same number of rows"):
        add_positional_embeddings(embedded_sequence, position_embedding_table)

def test_add_positional_empty_embedding():
    embedded_sequence = []    
    position_embedding_table = [[10.0, 10.0]]
    with pytest.raises(ValueError, match="Embedded sequence must be a non-empty list of lists"):
        add_positional_embeddings(embedded_sequence, position_embedding_table)

def test_add_positional_empty_position_table():
    embedded_sequence = [[0.5, 0.6]]
    position_embedding_table = []
    with pytest.raises(ValueError, match="Position embedding table must be a non-empty list of lists"):
        add_positional_embeddings(embedded_sequence, position_embedding_table)

def test_add_positional_rectangular_column_mismatch():
    embedded_sequence = [[0.5, 0.6, 0.7]]
    position_embedding_table = [[10.0, 10.0]]
    with pytest.raises(ValueError, match="Number of columns in the embedded sequence must equal the number of columns in the position embedding table"):
        add_positional_embeddings(embedded_sequence, position_embedding_table)








