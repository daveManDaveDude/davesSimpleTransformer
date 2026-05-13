from math import exp


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

def matrix_multiply(matrix1: list[list[int]], matrix2: list[list[int]]) -> list[list[int]]:
    if not matrix1 or len(matrix1) == 0 or not matrix2 or len(matrix2) == 0:
        raise ValueError("Matrices must be non-empty lists of lists")

    num_rows_matrix1 = len(matrix1)
    num_cols_matrix1 = len(matrix1[0])
    num_rows_matrix2 = len(matrix2)
    num_cols_matrix2 = len(matrix2[0])
    
    return_matrix = []

    for i in range(num_rows_matrix1):
        if len(matrix1[i]) != num_cols_matrix1:
            raise ValueError("All rows in the first matrix must have the same number of columns")

    for i in range(num_rows_matrix2):
        if len(matrix2[i]) != num_cols_matrix2:
            raise ValueError("All rows in the second matrix must have the same number of columns")

    if num_cols_matrix1 != num_rows_matrix2:
        raise ValueError("Number of columns in the first matrix must equal the number of rows in the second matrix")
    
    matrix2_transposed = transpose_matrix(matrix2)
    
    for i in range(num_rows_matrix1):
        return_row = []
        for j in range(len(matrix2_transposed)):
            dot_product_total = dot_product(matrix1[i], matrix2_transposed[j])
            return_row = return_row + [dot_product_total]
        return_matrix = return_matrix + [return_row]
        
    return return_matrix

def predict_from_logits(logits: list[float]) -> int:
    if len(logits) == 0:
        raise ValueError("Logits must be a non-empty list of floats")
    max_logit = max(logits)
    max_index = logits.index(max_logit) 
    return max_index

def predict_sequence(matrix1: list[list[float]]) -> list[int]:
    if not matrix1 or len(matrix1) == 0:
        raise ValueError("Matrix must be a non-empty list of lists")
    num_rows_matrix1 = len(matrix1)
    num_cols_matrix1 = len(matrix1[0])
    predictions = []
    for i in range(num_rows_matrix1):
        if len(matrix1[i]) != num_cols_matrix1:
            raise ValueError("All rows in the matrix must have the same number of columns")
        predictions = predictions + [predict_from_logits(matrix1[i])]
    return predictions

def softmax_from_logits(logits: list[float]) -> list[float]:
    if len(logits) == 0:
        raise ValueError("Logits must be a non-empty list of floats")
    total = 0
    return_probabilities = [0.0] * len(logits)
    for i in range(len(logits)):
        if not isinstance(logits[i], (int, float)):
            raise ValueError("All elements in logits must be numbers")
        return_probabilities[i] = exp(logits[i])
        total = total  + return_probabilities[i]
    for i in range(len(logits)):
        return_probabilities[i] = return_probabilities[i] / total
   
    return return_probabilities

def embed_token(token_id: int, embedding_matrix: list[list[float]]) -> list[float]:
    if not embedding_matrix or len(embedding_matrix) == 0:
        raise ValueError("Embedding matrix must be a non-empty list of lists")
    if not isinstance(token_id, int):
         raise ValueError("Token ID must be an integer")

    num_rows_embedding_matrix = len(embedding_matrix)
    num_cols_embedding_matrix = len(embedding_matrix[0])
    
    for i in range(num_rows_embedding_matrix):
        if len(embedding_matrix[i]) != num_cols_embedding_matrix:
            raise ValueError("All rows in the embedding matrix must have the same number of columns")
    
    if token_id < 0 or token_id >= num_rows_embedding_matrix:
        raise ValueError("Token ID must be a valid index in the embedding matrix")
    return embedding_matrix[token_id].copy()

def embed_sequence(token_ids: list[int], embedding_matrix: list[list[float]]) -> list[list[float]]:
    if not token_ids or len(token_ids) == 0:
        raise ValueError("Sequence must be a non-empty list of token IDs") 
    if embedding_matrix is None or len(embedding_matrix) == 0:
        raise ValueError("Embedding matrix must not be empty")   
    
    num_rows_embedding_matrix = len(embedding_matrix)
    embedded_sequence = []
    
    for token_id in token_ids:
        embedded_sequence = embedded_sequence + [embed_token(token_id, embedding_matrix)]
    return embedded_sequence

def embed_position(position_id, position_embedding_table):
    if not position_embedding_table or len(position_embedding_table) == 0:
        raise ValueError("Position embedding table must be a non-empty list of lists")
    if not isinstance(position_id, int):
         raise ValueError("Position ID must be an integer")

    num_rows_position_embedding_table = len(position_embedding_table)
    num_cols_position_embedding_table = len(position_embedding_table[0])
    
    for i in range(num_rows_position_embedding_table):
        if len(position_embedding_table[i]) != num_cols_position_embedding_table:
            raise ValueError("All rows in the position embedding table must have the same number of columns")
    
    if position_id < 0 or position_id >= num_rows_position_embedding_table:
        raise ValueError("Position ID must be a valid index in the position embedding table")
    return position_embedding_table[position_id].copy() 

def add_token_and_position(token_embedding: list[float], position_embedding: list[float]) -> list[float]:
    if not token_embedding or len(token_embedding) == 0:
        raise ValueError("Token embedding must be a non-empty list of floats")
    if not position_embedding or len(position_embedding) == 0:
        raise ValueError("Position embedding must be a non-empty list of floats")
    if len(token_embedding) != len(position_embedding):
        raise ValueError("Token embedding and position embedding must be of the same length")
    return add_vectors(token_embedding.copy(), position_embedding.copy())

def add_positional_embeddings(embedded_sequence: list[list[float]], position_embedding_table: list[list[float]]) -> list[list[float]]:
    if not embedded_sequence or len(embedded_sequence) == 0:
        raise ValueError("Embedded sequence must be a non-empty list of lists")
    if not position_embedding_table or len(position_embedding_table) == 0:
        raise ValueError("Position embedding table must be a non-empty list of lists")
    
    num_rows_embedded_sequence = len(embedded_sequence)
    num_cols_embedded_sequence = len(embedded_sequence[0])
    
    for i in range(num_rows_embedded_sequence):
        if len(embedded_sequence[i]) != num_cols_embedded_sequence:
            raise ValueError("All rows in the embedded sequence must have the same number of columns")
    
    num_rows_position_embedding_table = len(position_embedding_table)
    num_cols_position_embedding_table = len(position_embedding_table[0])
    
    for i in range(num_rows_position_embedding_table):
        if len(position_embedding_table[i]) != num_cols_position_embedding_table:
            raise ValueError("All rows in the position embedding table must have the same number of columns")
        
    if num_rows_position_embedding_table != num_rows_embedded_sequence:
        raise ValueError("Embedded sequence and position embedding table must have the same number of rows")   
    
    if num_cols_embedded_sequence != num_cols_position_embedding_table:
        raise ValueError("Number of columns in the embedded sequence must equal the number of columns in the position embedding table")
    
    return_sequence = []
    for i in range(num_rows_embedded_sequence):
        token_embedding = embedded_sequence[i]
        position_embedding = embed_position(i, position_embedding_table)
        return_sequence = return_sequence + [add_token_and_position(token_embedding, position_embedding)]
    
    return return_sequence.copy()
