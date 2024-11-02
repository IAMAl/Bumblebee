# Tiled Ring Buffer Attention
A memory-efficient implementation of attention mechanism combining ring buffers with hierarchical tiled computation. This implementation achieves efficient memory usage and computation through specialized optimizations for encoder-decoder transformers.


## Core Concepts
### Triple Ring Buffer System
The implementation uses three ring buffers for efficient memory management:

- Forward KV Cache: For inference and non-gradient operations
- Backward KV Cache: For gradient computation during training
- Decoder Feedback Buffer: For decoder self-attention output feedback

Each ring buffer operates with:

- Read and write pointers for active data window management
- Cache line aligned data width for efficient prefetching
- Sequential access patterns for optimal memory bandwidth
- Fixed size buffer with automatic pointer wraparound

### Ring Buffer Advantages

1. Hardware Efficiency

   - Cache-aligned data access
   - Efficient data prefetching
   - Sequential memory access patterns
   - Fixed memory footprint


2. KV Cache Management

   - Simple pointer-based management
   - Linear memory addressing
   - Automatic wraparound handling
   - Efficient encoder-decoder interaction


3. Decoder Optimizations

   - Selective matrix multiplication masking
   - Skip unnecessary data loading
   - Efficient right-shift operation
   - Sequential data arrangement



### Hierarchical Tiled Computation
Inspired by FlashAttention, our implementation uses a two-level tiling hierarchy:

1. Upper Level (Global Management)

   - Tile scheduling and coordination
   - Memory access pattern optimization
   - Inter-tile dependency handling


2. Lower Level (Processing)

   - Dense tiles: Full computation for valid regions
   - Triangle tiles: Special handling for diagonal blocks
   - Skip tiles: Automatic future token skipping


3. Tile Processing Orders

   - Q-major: Optimizes query tile reuse
   - K-major: Optimizes key/value tile reuse



## Implementation Details
### Basic Configuration
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
### Usage Examples

1. Model Initialization

```python
# Create model and move to GPU
model = RingBufferAttention(config).cuda()

# Set tile processing order
model.tiling_processor.tile_order = TileOrder.Q_MAJOR  # or K_MAJOR
```

2. Encoder Phase

```python
encoder_output, buffer_state = model(
    q=query,
    k=key,
    v=value,
    requires_grad=True
)
```

3. Decoder Phase

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
### Tiling Strategy Selection

1. Q-major Ordering (Default)
```python
Copymodel.tiling_processor.tile_order = TileOrder.Q_MAJOR
```
- Optimizes query tile reuse
- Preferred when query tiles > key tiles
- Reduces query access bandwidth


2. K-major Ordering
```python
model.tiling_processor.tile_order = TileOrder.K_MAJOR
```
- Optimizes key/value tile reuse
- Preferred when key tiles > query tiles
- Reduces key/value access bandwidth



## Performance Characteristics
### Space Complexity

- O(buffer_size * head_dim) per ring buffer
- Constant memory usage independent of sequence length
- Additional workspace proportional to tile sizes

### Time Complexity

- O(n * d) for n tokens and d head dimension
- Reduced memory bandwidth via tiling
- Efficient future token skipping
- Optional attention weight computation



## Requirements and Integration
### Dependencies

- PyTorch >= 1.8.0
- xFormers library
- CUDA-capable GPU (recommended)

## xFormers Compatibility

- Registered as standard attention module
- Follows xFormers factory patterns
- Compatible with xFormers configuration system

## Advanced Usage
### Training Mode
```python
# Initialize model for training
model = RingBufferAttention(RingBufferConfig())
model.tiling_processor.tile_order = TileOrder.Q_MAJOR
model.train()

# Forward pass with gradient computation
output, state = model(query, key, value, requires_grad=True)

# Decoder feedback step
decoder_output = model(
    dec_query, dec_key, dec_value,
    position=pos,
    is_decoder=True,
    buffer_state=state
)
```

### Inference Mode
```python
# Initialize model for inference
model = RingBufferAttention(RingBufferConfig())
model.eval()

# Forward pass without gradient computation
with torch.no_grad():
    output, state = model(query, key, value, requires_grad=False)
```

## Contributing
Areas for potential improvements:

- Advanced tile processing strategies
- Hardware-specific optimizations
- Auto-tuning systems
- Performance profiling tools
- Memory access pattern analysis

## Acknowledgments
This implementation builds upon:

- FlashAttention paper
- xFormers library
- Transformer optimization techniques

## License
- BSD 3-Clause License - See LICENSE file for details

## Contact
For questions and support, please open an issue on the GitHub repository.