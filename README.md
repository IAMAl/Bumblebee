# Tiled Ring Buffer Attention
A memory-efficient implementation of attention mechanism with ring buffers and hierarchical tiled computation. This implementation combines several optimization techniques:

## Features

- Triple Ring Buffer System

    - Forward KV Cache: For inference and non-gradient operations
    - Backward KV Cache: For gradient computation during training
    - Decoder Feedback Buffer: For decoder self-attention with output feedback


- Hierarchical Tiled Computation

    - Two-level tiling hierarchy:

        - Upper level: Global tile management and scheduling
        - Lower level: Specialized tile processing patterns


- Optimized tile processing orders:

    - Q-major: Maximizes query tile reuse
    - K-major: Maximizes key/value tile reuse


- Intelligent tile type identification:

    - Dense tiles: Full computation for non-causal regions
    - Triangle tiles: Special handling for diagonal blocks
    - Skip tiles: Automatic skipping of future tokens

## Triple Ring-Buffer System

The ring buffer is used in hardware in general in order to efficienty buffer the data. The ring buffer have two pointers; read pointer and write pointer. After reading or writing data, the pointer is incremented, respectively. When the value of the pointer achieved to "size of buffer - 1" the value is set to zero, namely the pointers make a window (scope) of active datum in the buffer.

The datum's width can be aligned with a line size of cache memory in host processor having data prefetch function. The ring-buffer has sequential access manner, so the prefetching works well with cache line alighnment. The host processor can efficiently provide datum to GPU without additional overheads.

KV cache can be implemented by the ring-buffer having fixed size. It is simply managed by the pointers. The encoder writes into the ring-buffer and then the write pointer may be updated. The decoder reads from the ring-buffer and then the read pointer may be updated. The ordered tensors make a sequential accesses to the ring-buffer, so the ring-buffer placed in linear addressing memory is sufficient, and utilized by the sequential accessings by the encoder and decoder.

The masking in decoder takes only a part of matrix for Q*K^T matrix multiplication, therefore the attention can skip the unnecessary data loadings and calculations. A ring-buffer for decoder's input from its output works for the right-shift for input, and masking taking only necessary part is supported by the ring-buffer's sequential data arrengement that has explicit data placement in the linear addressing memory.

## Hierarchical Tiled Computation with Ring-Buffer

We take idea of FlashAttention. FlashAttention takes tiling before softmax operation in attention. The the tensor is chunked to several blocks that can be fed into tile processsing. They use it for approximating the attention.  In stead, we use the idea for skipping unnecessary processings for tile-level in future vocablary in decoder.

Proposal attention also takes similar system, and works with a ring-buffer that is placed between output to input of decoder, a feedback path. The output ring-bufffer can be accessed with sequential manner in linear address space of memory, and the masking pattern is very simple, thus the reading from the ring-buffer is also simple as an incremental read-length.


## Memory Optimization Strategies

- Smart data reuse patterns
- Per-tile normalization for stability
- Configurable tile sizes for different hardware
- Efficient memory access patterns


## xFormers Integration

Code of proposal attention is compatible with xFormers and uses the xFormers factory system.

- Registered attention module
- Compatible with xFormers factory system
- Follows xFormers configuration patterns



## Usage
- Basic Configuration
```python
config = RingBufferConfig(
    # Required xFormers fields
    num_heads=8,
    head_dim=64,
    dropout=0.1,
    
    # Ring buffer configuration
    buffer_size=2048,      # Size of ring buffers
    block_size=128,        # Processing block size
    tile_size=32,          # Tile size for tiled attention
    head_tile_size=4,      # Number of heads to process together
    prefill_size=512,      # Size of prefill cache for skip calculation
    causal=True            # Whether to use causal masking
)
```
- Model Creation

```python
model = RingBufferAttention(config).cuda()
```

## Optionally specify tile processing order
```python
model.tiling_processor.tile_order = TileOrder.Q_MAJOR  # or K_MAJOR
```
- Forward Pass
```python
# Encoder phase
encoder_output, buffer_state = model(
    q=query,
    k=key,
    v=value,
    requires_grad=True
)
```

- Decoder phase
```python
decoder_output = model(
    q=decoder_query,
    k=decoder_key,
    v=decoder_value,
    position=current_pos,
    is_decoder=True,
    att_mask=attention_mask,          # Optional attention mask
    key_padding_mask=padding_mask,    # Optional padding mask
    needs_weights=False,              # Whether to return attention weights
    output_attentions=False           # Whether to return attention outputs
)
```

## Memory Optimization
### Tiling Strategies
The implementation supports two tile processing orders:

- Q-major Ordering (Default)

    - Optimizes for query tile reuse
    - Better when query tiles are larger than key tiles
    - Reduced memory bandwidth for query access
```python
model.tiling_processor.tile_order = TileOrder.Q_MAJOR
```

- K-major Ordering

    - Optimizes for key/value tile reuse
    - Better when key/value tiles are larger than query tiles
    - Reduced memory bandwidth for key/value access

```python
model.tiling_processor.tile_order = TileOrder.K_MAJOR
```

## Tile Configuration
Customize tile sizes based on your hardware:
```python
config = RingBufferConfig(
    # ... other configs ...
    tile_size=32,        # Size of Q/K tiles
    head_tile_size=4,    # Heads processed together
)
```

## Buffer Management

- Fixed memory footprint regardless of sequence length
- Efficient ring buffer rotation
- Automatic buffer state tracking

## Performance Characteristics
- Space Complexity

    - O(buffer_size * head_dim) per ring buffer
    - Constant memory usage regardless of sequence length
    - Additional tile workspace proportional to tile sizes

- Time Complexity

    - O(n * d) for n tokens and d head dimension
    - Reduced memory bandwidth through hierarchical tiling
    - Efficient skip calculation for future tokens
    - Optional attention weight computation when needed

## Requirements

- PyTorch >= 1.8.0
- xFormers library
- CUDA-capable GPU (recommended)

## Examples
- Training Example
```python
# Create model with specific tile order
model = RingBufferAttention(RingBufferConfig())
model.tiling_processor.tile_order = TileOrder.Q_MAJOR
model.train()
```
- Forward pass with gradient
```python
output, state = model(query, key, value, requires_grad=True)
```
- Forward pass without gradient
```python
with torch.no_grad():
    output, state = model(query, key, value, requires_grad=False)
```
-  Decoder step with feedback
```python
decoder_output = model(
    dec_query, dec_key, dec_value,
    position=pos,
    is_decoder=True,
    buffer_state=state
)
```
- Inference Example
```python
# Create model
model = RingBufferAttention(RingBufferConfig())
model.eval()
```


## Contributing
Contributions are welcome! Some areas for potential improvements:

Additional tile processing strategies
- Auto-tuning for tile sizes and ordering
- Hardware-specific optimizations
- Performance profiling tools
- Memory access pattern analysis

## License
BSD 3-Clause License - See LICENSE file for details

## Acknowledgments
This implementation draws inspiration from:

- FlashAttention paper
- xFormers library
- Various transformer optimization techniques

## Contact
For questions and support, please open an issue on the GitHub repository.