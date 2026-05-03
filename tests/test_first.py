import pytest

from tiny_transformer.first import SEQ_LEN, VOCAB_SIZE, reverse_tokens

def test_reverse_tokens_reverses_eight_digits():
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    target_tokens = reverse_tokens(input_tokens)

    assert target_tokens == [8, 7, 6, 5, 4, 3, 2, 1]

def test_vocab_size_equals_10():
    
    assert VOCAB_SIZE == 10

def test_seq_len_equals_8():
    
    assert SEQ_LEN == 8

def test_reverse_tokens_reverses_repeated_digits():
    input_tokens = [1, 1, 2, 2, 3, 3, 4, 4]

    target_tokens = reverse_tokens(input_tokens)

    assert target_tokens == [4, 4, 3, 3, 2, 2, 1, 1]

def test_reverse_tokens_reverses_less_than_eight_digits():
    input_tokens = [1, 2, 3, 4, 5, 6, 7]

    with pytest.raises(ValueError, match="Invalid input sequence"):
        reverse_tokens(input_tokens)

def test_reverse_tokens_fails_reverses_more_than_eight_digits():
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    with pytest.raises(ValueError, match="Invalid input sequence"):
        reverse_tokens(input_tokens)

def test_reverse_tokens_fails_reverses_non_integer_digits():
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8.7]

    with pytest.raises(ValueError, match="Invalid input sequence"):
        reverse_tokens(input_tokens)

def test_reverse_tokens_fails_reverses_float_digits():
    input_tokens = [1, 2, 3, 4, 5, 6, 7, 8.0]

    with pytest.raises(ValueError, match="Invalid input sequence"):
        reverse_tokens(input_tokens)