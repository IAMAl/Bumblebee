# Tiled Ring Buffer Attention
A memory-efficient implementation of attention mechanism with ring buffers and tiled computation. This implementation combines several optimization techniques:
## Features
1. Triple Ring Buffer System

- Forward KV Cache: For inference and non-gradient operations
- Backward KV Cache: For gradient computation during training
- Decoder Feedback Buffer: For decoder self-attention with output feedback

2. FlashAttention-style Tiling

- Query/Key/Head tiling for memory efficiency
- Block-based processing to optimize memory access
- Configurable tile sizes for different memory constraints

3. Skip Calculation System

- Tracks last calculated positions
- Skips unnecessary masking computations
- Configurable prefill size for efficiency

4. xFormers Integration

## Registered attention module
- Compatible with xFormers factory system
- Follows xFormers configuration patterns

## Usage
- Basic Configuration
```pythhon
config = RingBufferConfig(
    buffer_size=2048,      # Size of ring buffers
    block_size=128,        # Processing block size
    tile_size=32,          # Tile size for flash attention
    head_tile_size=4,      # Number of heads to process together
    prefill_size=512       # Size of prefill cache for skip calculation
)
```

## Model Creation
```python
model = CompleteRingBufferAttention(config).cuda()
```
## Forward Pass
```python
# Encoder phase
encoder_output, buffer_state = model(
    q=query,
    k=key,
    v=value,
    requires_grad=True
)
```
## Decoder phase
```python
decoder_output = model(
    q=decoder_query,
    k=decoder_key,
    v=decoder_value,
    position=current_pos,
    is_decoder=True,
    buffer_state=buffer_state
)
```
### Memory Optimization
- Tiling Parameters
- The implementation uses configurable tiling parameters:

    - tile_size: Size of query/key tiles
    - head_tile_size: Number of attention heads processed together
    - block_size: Size of memory blocks in ring buffers

### Buffer Management

- Fixed memory footprint regardless of sequence length
- Efficient handling of long sequences
- Automatic buffer rotation and reuse

### Performance Characteristics
- Space Complexity

    - O(buffer_size * head_dim) per ring buffer
    - Constant memory usage regardless of sequence length

- Time Complexity

    - O(n * d) for n tokens and d head dimension
    - Reduced memory bandwidth through tiling
    - Efficient skip calculation for decoder

## Requirements

- PyTorch >= 1.8.0
- xFormers library
- CUDA-capable GPU (recommended)


## Examples
- Training Example
```python
# Create model
model = RingBufferAttention(RingBufferConfig())
model.train()
```

### Forward pass with gradient
```python
output, state = model(query, key, value, requires_grad=True)
```
### Decoder step with feedback
```python
decoder_output = model(
    dec_query, dec_key, dec_value,
    position=pos,
    is_decoder=True,
    buffer_state=state
)
```
### Inference Example
```python
# Create model
model = CompleteRingBufferAttention(RingBufferConfig())
model.eval()
```

### Forward pass without gradient
```python
with torch.no_grad():
    output, state = model(query, key, value, requires_grad=False)
```

## Contributing
Contributions are welcome! Some areas for potential improvements:

- Additional tiling strategies
- More skip calculation optimizations
- Enhanced buffer management techniques
- Performance benchmarks and optimizations

## License
BSD 3-Clause License - See LICENSE file for details

## Acknowledgments
This implementation draws inspiration from:

- FlashAttention paper
- xFormers library
- Various transformer optimization techniques

## Contact
For questions and support, please open an issue on the GitHub repository.
