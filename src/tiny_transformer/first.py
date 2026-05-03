VOCAB_SIZE = 10
SEQ_LEN = 8

input_seq = [1, 2, 3, 4, 5, 6, 7, 8]

def is_valid(seq):
    if len(seq) != SEQ_LEN:
        return False
    for token in seq:
        if type(token) != int:  
            return False
        if token < 0 or token >= VOCAB_SIZE:
            return False
    return True

def reverse(seq):
    return_seq = []
    for token in range(SEQ_LEN -1, -1, -1):
        return_seq = return_seq + [seq[token]]
    return return_seq


def reverse_tokens(input_seq):
    if not is_valid(input_seq):
        raise ValueError("Invalid input sequence")
    return reverse(input_seq)    


print(reverse_tokens(input_seq))
print(reverse_tokens([0, 0, 1, 2, 3, 4, 2,5]))