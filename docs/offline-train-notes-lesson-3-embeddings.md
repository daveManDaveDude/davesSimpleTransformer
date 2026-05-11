# Offline Notes: Tiny Transformer Progress And Next Steps

These notes are for working offline on the train. They summarize how the project got here, what is currently working, what still needs review, and the next small coding steps.

The project goal is to build a tiny transformer from scratch in plain Python that learns to reverse 8 digit tokens:

```text
input:  1 2 3 4 5 6 7 8
target: 8 7 6 5 4 3 2 1
```

The teaching rule still applies: do not rush to a complete transformer. Build one small, inspectable piece at a time.

## Where You Started

Lesson 1 established the task:

- Vocabulary size is `10`, for digits `0` through `9`.
- Sequence length is `8`.
- A valid input is exactly 8 integer tokens.
- The target is the input sequence reversed.

The important idea from Lesson 1 was that a token is just an integer ID. At this stage, `3` means "the digit token 3", not a model vector yet.

## Lesson 2: Vector And Matrix Utilities

You then built the plain-Python math helpers that the transformer will need later.

Current utilities in `src/tiny_transformer/vector.py`:

```text
add_vectors
dot_product
transpose_matrix
matrix_vector_multiply
matrix_multiply
```

The main concept from this lesson was shape checking.

Examples:

```text
vector + vector:
[3] + [3] -> [3]

dot product:
[3] dot [3] -> scalar

matrix-vector multiply:
[2, 3] @ [3] -> [2]

matrix-matrix multiply:
[2, 3] @ [3, 2] -> [2, 2]
```

The important habit you developed was:

```text
1. Check the object is non-empty.
2. Check the matrix is rectangular.
3. Check the inner dimensions match.
4. Only then do the calculation.
```

That habit will matter all the way through attention.

## Lesson 3: Logits, Softmax, And Prediction

This lesson introduced model scores.

A logit is a raw score. It is not a probability.

For a small vocabulary:

```text
logits = [1.0, 5.0, 2.0]
```

The prediction is the index of the largest score:

```text
largest value = 5.0
index = 1
prediction = 1
```

You implemented:

```text
predict_from_logits
predict_sequence
softmax_from_logits
```

Softmax turns raw scores into probabilities:

```text
logits -> exp scores -> divide by total -> probabilities
```

For example:

```text
softmax([1.0, 2.0, 3.0])
approximately equals
[0.0900, 0.2447, 0.6652]
```

The checks you added are the right kind:

- output length matches input length
- probabilities are between `0` and `1`
- probabilities sum to about `1.0`
- largest logit gets largest probability
- input logits are not mutated

This is a good place to be.

## Current Lesson: Token Embeddings

You have started embedding helpers:

```text
embed_token
embed_sequence
```

The concept is:

```text
token ID -> embedding vector
```

An embedding table is just a matrix.

Small example:

```text
embedding_table = [
  [0.1, 0.2],  # token 0
  [0.3, 0.4],  # token 1
  [0.5, 0.6],  # token 2
]
```

Then:

```text
embed_token(0, table) -> [0.1, 0.2]
embed_token(2, table) -> [0.5, 0.6]
```

For a sequence:

```text
tokens = [2, 0, 1]
```

The embedded sequence should be:

```text
[
  [0.5, 0.6],
  [0.1, 0.2],
  [0.3, 0.4],
]
```

Shape tracing:

```text
tokens:          [3]
embedding table: [3, 2]
result:          [3, 2]
```

For the real tiny transformer:

```text
tokens:          [8]
embedding table: [10, 16]
result:          [8, 16]
```

## Current Test Status

At the time these notes were written, the current tests pass:

```text
48 passed
```

Command used:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest .\tests\test_first.py .\tests\test_vectors.py -q
```

There is also a pytest cache warning about `.pytest_cache` permissions. That warning is not about your transformer code.

## Review Of Current Embedding Work

What looks good:

- `embed_token` checks the embedding matrix is non-empty.
- `embed_token` checks the embedding matrix is rectangular.
- `embed_token` checks token IDs are in range.
- `embed_token` returns a copy of the selected row, which protects the embedding table from accidental mutation.
- You started tests for token lookup, invalid token IDs, empty tables, jagged tables, empty sequences, and copy behavior.

Things to fix next:

1. Add a token type check.

Right now the intended behavior says token IDs should be integers. A token like `1.5` should raise a clear `ValueError`.

Think:

```text
token_id must be an int
token_id must be >= 0
token_id must be < number of rows in embedding table
```

2. Fix the duplicate test name.

There are two tests named:

```text
test_embed_token_invalid_token_id
```

In Python, the second one replaces the first one during test collection. Rename one of them so both tests are visible.

3. Make the sequence test call `embed_sequence`.

The current sequence test builds the result with:

```text
[embed_token(token_id, table) for token_id in token_ids]
```

That tests `embed_token` again, not `embed_sequence`.

The test should directly exercise:

```text
embed_sequence([2, 0, 1], table)
```

Expected result:

```text
[[0.5, 0.6], [0.1, 0.2], [0.3, 0.4]]
```

4. Make `embed_sequence` return independent row copies.

`embed_token` returns a copied row. `embed_sequence` should behave the same way for each embedded token.

Reason:

```text
result = embed_sequence([1], table)
result[0][0] = 999
table[1][0] should still be 0.3
```

This protects the embedding table.

5. Let `embed_sequence` reuse `embed_token`.

Conceptually:

```text
embed_sequence = call embed_token once for each token
```

This keeps validation behavior consistent.

Do not over-abstract it. Just notice that the per-token work already exists.

## Train-Friendly Coding Checklist

Work through these one by one. Do not jump ahead.

### Step 1: Clean Up Test Names

Find the duplicate:

```text
test_embed_token_invalid_token_id
```

Rename the second one to something specific, such as:

```text
test_embed_token_rejects_out_of_range_token_id
```

or:

```text
test_embed_token_rejects_negative_and_too_large_token_id
```

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest .\tests\test_vectors.py -q
```

### Step 2: Add The Float Token Test

Uncomment or add a test for:

```text
embed_token(1.5, table)
```

Expected behavior:

```text
ValueError
```

Suggested message:

```text
Token ID must be an integer
```

Only after the test fails, update `embed_token`.

Conceptual check order:

```text
1. embedding matrix is non-empty
2. embedding matrix is rectangular
3. token_id is an int
4. token_id is in range
5. return a copy of the row
```

### Step 3: Make The Sequence Test Use `embed_sequence`

Change the sequence test so it calls:

```text
embed_sequence(token_ids, table)
```

not a list comprehension around `embed_token`.

Expected result:

```text
[[0.5, 0.6], [0.1, 0.2], [0.3, 0.4]]
```

This verifies the function you are actually trying to build.

### Step 4: Add `embed_sequence` Token Type Tests

Add tests for:

```text
embed_sequence([0, 1.5], table)
embed_sequence([0, -1], table)
embed_sequence([0, 3], table)
```

Expected behavior:

```text
ValueError
```

Keep the messages simple and consistent with `embed_token`.

### Step 5: Add `embed_sequence` Copy Test

Use this idea:

```text
table = [
  [0.1, 0.2],
  [0.3, 0.4],
  [0.5, 0.6],
]

result = embed_sequence([1], table)
result[0][0] = 999

table[1] should still be [0.3, 0.4]
```

This test teaches an important Python list lesson: copying the outer list is not enough if the inner row lists are shared.

### Step 6: Refactor Only If The Tests Guide You

After the tests exist, consider simplifying `embed_sequence` so it uses `embed_token` for each token.

Pseudocode only:

```text
check sequence is non-empty
embedded_sequence = []
for each token_id:
    append embed_token(token_id, embedding_matrix)
return embedded_sequence
```

Because `embed_token` already validates the table, this may repeat validation for each token. That is okay for now. Clarity matters more than speed in this project.

## Important Shape Questions To Ask Yourself

For each embedding function, ask:

```text
What shape goes in?
What shape comes out?
What invalid shapes should fail early?
```

For `embed_token`:

```text
token_id: scalar int
embedding table: [vocab_size, model_dim]
result: [model_dim]
```

For `embed_sequence`:

```text
token_ids: [sequence_len]
embedding table: [vocab_size, model_dim]
result: [sequence_len, model_dim]
```

For this project:

```text
token_ids: [8]
embedding table: [10, 16]
result: [8, 16]
```

## What Comes After Embeddings

Once token embeddings are solid, the next lesson is positional embeddings.

Why position matters:

```text
[1, 2, 3, 4, 5, 6, 7, 8]
```

To reverse the sequence, the model must know not only the digit value but where the digit appeared.

Token embedding says:

```text
what digit is this?
```

Position embedding says:

```text
where is this digit?
```

Later, you will combine them:

```text
token vector + position vector
```

Shape:

```text
token embeddings:    [8, 16]
position embeddings: [8, 16]
combined:            [8, 16]
```

Do not start that until `embed_token` and `embed_sequence` feel boring and reliable.

## Quick Review Prompt For Offline Use

When you come back online, paste this:

```text
I worked offline on the embedding functions.

Current lesson:
Token embeddings.

Please review:
- embed_token
- embed_sequence
- the embedding tests

Check especially:
- token type validation
- out-of-range token IDs
- jagged embedding tables
- whether returned embeddings accidentally mutate the table
- whether my tests actually call the functions they claim to test

Do not rewrite the solution. Review and suggest the smallest next step.
```

