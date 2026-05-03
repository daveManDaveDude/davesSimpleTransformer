# Tiny Transformer Learning Plan — Coaching Mode

This repo is for learning by building a tiny transformer yourself.

The project is based on an educational transformer that reverses an 8-digit sequence:

```text
input:  1 2 3 4 5 6 7 8
target: 8 7 6 5 4 3 2 1
```

The intended architecture is deliberately tiny:

| Item | Value |
|---|---:|
| Vocabulary | 10 digits, `0` to `9` |
| Sequence length | 8 |
| Model dimension | 16 |
| Layers | 1 |
| Attention heads | 1 |
| Main task | Predict the reversed sequence |

The point is **not** to get a finished implementation quickly. The point is to learn the pieces of a transformer by building them gradually.

---

## 1. Coaching Rules

When using ChatGPT or Codex with this repo, the assistant should act as a tutor and reviewer, not as someone completing the project for me.

### The assistant should do this

- Explain the current concept clearly.
- Ask short guiding questions when useful.
- Review code I have already written.
- Point out bugs, shape mismatches, and design issues.
- Suggest what file, function, test, or concept I should inspect next.
- Give small examples using tiny numbers where needed.
- Give pseudocode only when it helps understanding.
- Help me write or improve tests.
- Help me debug errors I bring to the session.
- Keep the current lesson small enough to finish in about an hour.

### The assistant should not do this

- Do not give me a complete code solution.
- Do not implement the whole lesson for me unless I explicitly ask.
- Do not jump ahead to later lessons.
- Do not introduce PyTorch, TensorFlow, JAX, Hugging Face, or scikit-learn during the core build.
- Do not hide the maths behind library calls.
- Do not refactor unrelated parts of the repo.
- Do not turn this into a production ML project.
- Do not optimise before the code is understandable.

### Code policy

By default, provide:

```text
explanation -> hints -> review -> suggested next small step
```

Only provide actual code when I explicitly ask for it, for example:

```text
I am stuck. Give me a small code example for this one function only.
```

Even then, keep the code limited to the smallest useful example.

---

## 2. Suggested Repo Structure

```text
repo-root/
  AGENTS.md
  docs/
    tiny-transformer-learning-plan.md
  src/
    tiny_transformer/
      __init__.py
      data.py
      tensor.py
      layers.py
      model.py
      train.py
      infer.py
  tests/
    test_data.py
    test_tensor.py
    test_layers.py
    test_model.py
  README.md
```

---

## 3. Standard Prompt Prefix for Each Session

Use this at the start of a ChatGPT or Codex session:

```text
I am building a tiny transformer from scratch to learn how transformers work.

Important coaching rule:
Do not give me the full code solution. Act as a tutor. Give explanations, hints, review of my attempt, debugging guidance, and suggestions for what to inspect next. Only give code if I explicitly ask for a small example or a specific function.

Current lesson:
[PASTE LESSON NAME HERE]

Current repo state or my attempt:
[PASTE RELEVANT CODE / ERROR / NOTES HERE]
```

For Codex, add:

```text
Do not modify files unless I explicitly ask you to. First review, explain, and suggest the smallest next step. If I ask you to edit files, make the smallest safe change and include tests for only that change.
```

---

## 4. Lesson 1 — Define the Reversal Task

### Goal

Represent the problem before building any model.

```text
input:  [1, 2, 3, 4, 5, 6, 7, 8]
target: [8, 7, 6, 5, 4, 3, 2, 1]
```

### Concepts

- Token IDs
- Vocabulary
- Sequence length
- Input/target pairs
- Synthetic data

### Your task

Create simple data helpers for:

- generating one random 8-digit sequence
- reversing a sequence
- generating a small set of training examples

### Prompt

```text
Lesson 1: dataset layer.

Help me understand how to represent the 8-digit reversal task.

Do not write the full solution. Explain the concept, then give me hints for the functions I should create. Review my attempt if I paste it. Suggest tests I should write.
```

---

## 5. Lesson 2 — Basic Vector and Matrix Utilities

### Goal

Build just enough maths helpers to support the toy transformer.

### Concepts

- Vector
- Matrix
- Shape
- Dot product
- Transpose
- Matrix-vector multiplication
- Matrix-matrix multiplication

### Your task

Implement small plain-Python helpers using lists.

### Prompt

```text
Lesson 2: vector and matrix utilities.

Help me understand the minimum vector and matrix operations needed for the tiny transformer.

Do not write the full code. Explain the operations using tiny examples. Give hints about shape checks and tests. Review my implementation if I paste it.
```

---

## 6. Lesson 3 — Logits, Softmax, and Prediction

### Goal

Understand how raw scores become probabilities.

### Concepts

- Logits
- Softmax
- Probability distribution
- Argmax
- Numerical stability

### Your task

Create helpers for:

- stable softmax
- argmax
- predicting one token from logits

### Prompt

```text
Lesson 3: logits, softmax, and prediction.

Explain logits and softmax for a 10-digit vocabulary.

Do not write the full implementation. Give me the mathematical idea, hints, and edge cases. Suggest tests. Review my attempt if I paste it.
```

---

## 7. Lesson 4 — Cross-Entropy Loss

### Goal

Measure how wrong the model's predictions are.

### Concepts

- Target class
- Negative log likelihood
- Cross-entropy
- Per-token loss
- Sequence loss

### Your task

Implement:

- cross-entropy for one token
- average cross-entropy for a sequence

### Prompt

```text
Lesson 4: cross-entropy loss.

Help me understand how to measure prediction error for digit classification.

Do not give me the finished code. Explain the formula, give a tiny numeric example, suggest tests, and review my attempt.
```

---

## 8. Lesson 5 — Token Embeddings

### Goal

Turn digit IDs into learned vectors.

### Concepts

- Embedding table
- Vocabulary size
- Model dimension
- Lookup
- Trainable parameters

### Your task

Create:

- an embedding table of shape `vocab_size x model_dim`
- lookup for one token
- lookup for a sequence

### Prompt

```text
Lesson 5: token embeddings.

Help me understand why token IDs need embeddings and what shape the embedding table should have.

Do not implement it for me. Give hints, shape guidance, and test ideas. Review my code if I paste it.
```

---

## 9. Lesson 6 — Positional Information

### Goal

Give the model information about token positions.

### Concepts

- Position
- Learned positional embedding
- Adding token and position vectors
- Why reversal requires position awareness

### Your task

Create:

- a positional embedding table of shape `sequence_len x model_dim`
- a function to add token and position embeddings

### Prompt

```text
Lesson 6: positional embeddings.

Explain why self-attention needs position information for an 8-digit reversal task.

Do not write the full code. Give hints about shapes and addition. Suggest tests. Review my attempt.
```

---

## 10. Lesson 7 — Linear Layers

### Goal

Build a reusable learned projection.

### Concepts

- Weight matrix
- Bias vector
- Input dimension
- Output dimension
- Projection
- Parameters

### Your task

Create a plain-Python linear layer that can later be used for Q, K, V, and output projection.

### Prompt

```text
Lesson 7: linear layers.

Help me understand a linear layer as y = xW + b.

Do not give the full implementation. Explain the shape of W and b. Give hints for one-vector and sequence-vector versions. Suggest tests and review my code.
```

---

## 11. Lesson 8 — Query, Key, and Value Projections

### Goal

Prepare embedded vectors for attention.

### Concepts

- Query
- Key
- Value
- Learned projection
- Self-attention preparation

### Your task

Use linear layers to produce Q, K, and V from the embedded sequence.

### Prompt

```text
Lesson 8: Q, K, and V projections.

Explain queries, keys, and values for one self-attention head.

Do not implement the full code. Give shape guidance and hints. Review my attempt and suggest what to check if the shapes are wrong.
```

---

## 12. Lesson 9 — Attention Scores

### Goal

Compute how strongly each position should attend to each other position.

### Concepts

- Dot product similarity
- QKᵀ
- Score matrix
- Scaling by sqrt(head_dim)
- Shape `sequence_len x sequence_len`

### Your task

Compute attention scores from Q and K.

### Prompt

```text
Lesson 9: attention scores.

Help me understand QK^T and why it creates an 8x8 score matrix.

Do not give the final function. Give a tiny hand-worked example, hints, and tests. Review my code if I paste it.
```

---

## 13. Lesson 10 — Attention Weights

### Goal

Turn attention scores into probabilities.

### Concepts

- Row-wise softmax
- Attention distribution
- One output position attending to input positions

### Your task

Apply softmax to each row of the score matrix.

### Prompt

```text
Lesson 10: attention weights.

Explain why softmax is applied row-by-row to the attention score matrix.

Do not write the full solution. Give hints, edge cases, and tests that verify each row sums to 1. Review my attempt.
```

---

## 14. Lesson 11 — Apply Attention to Values

### Goal

Use attention weights to blend value vectors.

### Concepts

- Weighted sum
- Context vector
- weights x V
- Copying information between positions

### Your task

Create the attention output by multiplying attention weights by V.

### Prompt

```text
Lesson 11: applying attention to values.

Explain how attention weights blend value vectors.

Do not implement it for me. Give a small numeric example, shape hints, and test ideas. Review my code if I paste it.
```

---

## 15. Lesson 12 — Single-Head Self-Attention Forward Pass

### Goal

Combine Q, K, V, scores, weights, and value blending.

### Concepts

- Self-attention
- One head
- Attention matrix
- End-to-end attention flow

### Your task

Create a forward pass for one self-attention head.

### Prompt

```text
Lesson 12: single-head self-attention.

Help me combine the pieces I already wrote into a self-attention forward pass.

Do not write the whole implementation. Help me plan the function boundaries, trace shapes, and review my attempt.
```

---

## 16. Lesson 13 — Residual Connection

### Goal

Add the attention output back to the original input vectors.

### Concepts

- Residual connection
- Shape compatibility
- Preserving information
- Easier optimisation

### Your task

Add residual vector addition after attention.

### Prompt

```text
Lesson 13: residual connection.

Explain what a residual connection does in this tiny transformer.

Do not code it for me. Give hints about shape checks and tests. Review my implementation if I paste it.
```

---

## 17. Lesson 14 — Projection Back to Vocabulary

### Goal

Convert model vectors back into digit logits.

### Concepts

- Output projection
- Vocabulary logits
- Per-position prediction
- Shape `sequence_len x vocab_size`

### Your task

Project each model vector of size 16 to 10 logits.

### Prompt

```text
Lesson 14: output projection.

Explain why each sequence position needs 10 logits.

Do not write the finished code. Give shape guidance, test ideas, and review my attempt.
```

---

## 18. Lesson 15 — Full Forward Pass

### Goal

Assemble the model from input tokens to output logits.

### Concepts

- Model composition
- Parameters
- Forward pass
- Shape tracing

### Your task

Build the forward-only tiny transformer:

```text
tokens -> token embedding + positional embedding -> self-attention -> residual -> projection -> logits
```

### Prompt

```text
Lesson 15: full forward pass.

Help me assemble the pieces into a forward pass.

Do not write the whole model for me. Help me trace shapes, identify parameters, design tests, and review my attempt.
```

---

## 19. Lesson 16 — Parameter Counting

### Goal

Count trainable parameters and compare the result with the reference diagram.

### Concepts

- Trainable parameters
- Embedding parameter count
- Linear layer parameter count
- Biases
- Architecture differences

### Your task

Create a parameter-counting breakdown.

### Prompt

```text
Lesson 16: parameter counting.

Help me count the trainable parameters in my tiny transformer and compare the count with the reference diagram's 1,216 parameters.

Do not force the architecture to match. Help me reason about which tables, weights, and biases are included. Review my count.
```

---

## 20. Lesson 17 — Numerical Gradients

### Goal

Understand gradients before building autograd.

### Concepts

- Gradient
- Finite differences
- Loss surface
- Learning rate
- Gradient descent

### Your task

Estimate the gradient of a simple scalar function, then optionally one model parameter.

### Prompt

```text
Lesson 17: numerical gradients.

Explain gradients using finite differences and a tiny scalar example.

Do not build training for me. Give hints, checks, and review my experiment.
```

---

## 21. Lesson 18 — Minimal Autograd Engine

### Goal

Build scalar automatic differentiation.

### Concepts

- Computational graph
- Chain rule
- Backward pass
- Scalar values
- Micrograd-style learning

### Your task

Create a tiny scalar `Value` type with enough operations to support learning.

### Prompt

```text
Lesson 18: minimal autograd.

Help me design a tiny scalar autograd engine.

Do not give me the finished implementation. Explain the graph and chain rule. Give hints for the operations I need and tests for simple derivatives. Review my attempt.
```

---

## 22. Lesson 19 — Autograd-Compatible Layers

### Goal

Make parameters differentiable.

### Concepts

- Parameters as differentiable values
- Gradients on weights
- Zeroing gradients
- Collecting parameters

### Your task

Adapt embeddings and linear layers so parameters can receive gradients.

### Prompt

```text
Lesson 19: autograd-compatible layers.

Help me adapt my existing layers to use my Value objects.

Do not rewrite everything. Give a migration plan, shape guidance, tests, and review my changes.
```

---

## 23. Lesson 20 — Differentiable Loss

### Goal

Connect cross-entropy loss to logits through autograd.

### Concepts

- Differentiable softmax or log-softmax
- Cross-entropy
- Gradients flowing to logits

### Your task

Make sequence loss differentiable.

### Prompt

```text
Lesson 20: differentiable loss.

Help me make cross-entropy work with my autograd engine.

Do not give me the finished code. Explain the operations needed, suggest tests that prove gradients flow, and review my attempt.
```

---

## 24. Lesson 21 — First Training Loop

### Goal

Overfit one example to prove learning works.

### Concepts

- Forward pass
- Loss
- Backward pass
- Zero gradients
- Parameter update
- Learning rate

### Your task

Train on one fixed example and show loss decreasing.

### Prompt

```text
Lesson 21: first training loop.

Help me understand and build the first training loop by overfitting one fixed 8-digit sequence.

Do not write the complete training script. Give the loop structure as guidance, help me debug my implementation, and suggest checks.
```

---

## 25. Lesson 22 — Random Training Examples

### Goal

Train on many randomly generated reversal examples.

### Concepts

- Synthetic training data
- Generalisation
- Token accuracy
- Full-sequence accuracy
- Evaluation

### Your task

Train and evaluate on random examples.

### Prompt

```text
Lesson 22: random-example training.

Help me extend training from one example to random reversal examples.

Do not implement it all for me. Help me design accuracy metrics, evaluation checks, and debugging steps if the model does not learn.
```

---

## 26. Lesson 23 — Inspect Attention

### Goal

Look inside the trained model.

### Concepts

- Attention map
- Reverse diagonal pattern
- Interpretability
- Why attention may not be perfectly clean

### Your task

Return and print the 8x8 attention matrix.

### Prompt

```text
Lesson 23: inspecting attention.

Help me inspect the attention weights of my trained model.

Do not write a full visualisation system. Suggest a simple text-based inspection approach and review my output.
```

---

## 27. Lesson 24 — Save and Load Model State

### Goal

Persist learned parameters.

### Concepts

- Model state
- Serialisation
- JSON
- Reproducibility
- Same prediction after reload

### Your task

Save and load parameters.

### Prompt

```text
Lesson 24: save and load.

Help me design an inspectable save/load format for the tiny model.

Do not write the full implementation. Give hints, risks, and tests to prove predictions match after reload.
```

---

## 28. Lesson 25 — Command-Line Inference

### Goal

Run the trained model from the terminal.

### Concepts

- Inference
- CLI arguments
- Input validation
- Comparing prediction with expected reversal

### Your task

Create a simple inference command.

### Prompt

```text
Lesson 25: command-line inference.

Help me design a small CLI for running the trained model on exactly 8 digits.

Do not give the complete script. Give input-validation guidance, test ideas, and review my attempt.
```

---

## 29. Lesson 26 — README and Learning Notes

### Goal

Document what the project does and what you learned.

### Concepts

- Reproducibility
- Architecture summary
- Running tests
- Running training
- Limitations

### Your task

Write a README that reflects the real current state of the repo.

### Prompt

```text
Lesson 26: README and learning notes.

Help me improve the README for the current repo.

Do not invent features. Review what exists, suggest structure, and help me explain the project clearly.
```

---

## 30. Optional Lesson 27 — NumPy Comparison

### Goal

Compare plain Python with NumPy after the educational version works.

### Concepts

- Vectorisation
- Array shapes
- Performance
- Same maths, faster operations

### Prompt

```text
Optional lesson: NumPy comparison.

Now that the plain-Python version works, help me compare it with a NumPy version.

Do not replace the educational code. Explain what NumPy changes and suggest a small comparison plan.
```

---

## 31. Optional Lesson 28 — PyTorch Comparison

### Goal

Compare the educational build with a framework version.

### Concepts

- Framework autograd
- `nn.Module`
- Optimisers
- Tensor operations

### Prompt

```text
Optional lesson: PyTorch comparison.

Now that the from-scratch version works, help me build a tiny PyTorch equivalent for comparison.

Keep it close to my architecture and explain what PyTorch is doing for me.
```

---

## 32. Milestones

### Milestone A — Data and Maths Foundation

Complete lessons 1 to 4.

You should be able to:

- Generate reversal examples.
- Manipulate vectors and matrices.
- Convert logits to probabilities.
- Calculate loss.

### Milestone B — Forward-Only Transformer

Complete lessons 5 to 16.

You should be able to:

- Run an untrained forward pass.
- Trace every shape.
- Count parameters.
- Explain each part of the data flow.

### Milestone C — Learning

Complete lessons 17 to 22.

You should be able to:

- Explain gradients.
- Use your own tiny autograd engine.
- Train the model.
- Show loss decreasing.
- Measure accuracy.

### Milestone D — Inspection and Usability

Complete lessons 23 to 26.

You should be able to:

- Inspect attention.
- Save and load model state.
- Run inference from the terminal.
- Explain the project in the README.

### Milestone E — Framework Comparison

Complete optional lessons 27 and 28 only after the plain-Python version works.

---

## 33. Design Questions to Revisit Later

Do not answer these at the start.

1. Does the model need positional embeddings to solve reversal?
2. Does one attention head learn a clear reverse diagonal?
3. Does the residual connection help this tiny task?
4. Does the parameter count match 1,216?
5. Which architecture choices affect the count?
6. Can the model generalise to unseen digit sequences?
7. Can it generalise to longer sequences?
8. What changes if we add start/end tokens?
9. What changes if we train autoregressively?
10. What changes if we add more heads or layers?

---

## 34. First Prompt to Use

```text
I want to begin Lesson 1 from my tiny transformer learning plan.

I am building a tiny transformer from scratch in plain Python to learn how transformers work. The task is: input 8 digit tokens and predict the reversed sequence.

Important coaching rule:
Do not give me the full code solution. Act as a tutor. Give explanations, hints, review of my attempt, debugging guidance, and suggestions for what to inspect next. Only give code if I explicitly ask for a small example or a specific function.

For this session, help me understand and plan only the dataset layer.

Constraints:
- Vocabulary is digits 0 to 9.
- Sequence length is 8.
- Use plain Python only.
- Do not build the model yet.
- Do not use AI/ML frameworks.
- Keep the code beginner-readable.
- Suggest tests I should write.
```
