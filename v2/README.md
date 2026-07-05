# SML v2

## Updates

- Optimize the model for MLX because the training on PyTorch+MPS was very slow.
- Remove attention dropout to use MLX fast attention.