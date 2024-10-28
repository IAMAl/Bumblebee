import torch
import torch.nn as nn
from typing import Optional, Tuple, NamedTuple
from dataclasses import dataclass
from xformers.components import Attention, AttentionConfig
from xformers.components.attention import register_attention
import math

@dataclass
class RingBufferConfig(AttentionConfig):
    """Configuration for ring buffer system"""
    buffer_size: int = 2048  # KV cache size
    block_size: int = 128    # Processing block size
    tile_size: int = 32      # Tile size for flash attention
    head_tile_size: int = 4  # Number of heads to process together
    prefill_size: int = 512  # Size of prefill cache for skip calculation

class BufferState(NamedTuple):
    """Track states of buffers and positions"""
    kv_fwd_pos: int    # Forward KV cache position
    kv_bwd_pos: int    # Backward KV cache position
    dec_pos: int       # Decoder feedback position
    last_calc: int     # Last calculated position for skip mask

class TileConfig(NamedTuple):
    """Tiling configuration for flash attention"""
    query_tiles: int   # Number of query tiles
    key_tiles: int     # Number of key tiles
    head_tiles: int    # Number of head tiles
    block_tiles: int   # Number of block tiles
    sizes: Tuple[int, int, int, int]  # Tile sizes (q, k, h, b)

class RingBuffer:
    """Base ring buffer implementation"""
    def __init__(self, 
                buffer_size: int, 
                num_heads: int, 
                head_dim: int,
                block_size: int):
        self.buffer_size = buffer_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = (buffer_size + block_size - 1) // block_size
        
        # Initialize buffer
        self.key_buf = nn.Parameter(
            torch.zeros(self.num_blocks, block_size, num_heads, head_dim),
            requires_grad=False
        )
        self.value_buf = nn.Parameter(
            torch.zeros_like(self.key_buf),
            requires_grad=False
        )
        self.valid_mask = nn.Parameter(
            torch.zeros(self.num_blocks, block_size, dtype=torch.bool),
            requires_grad=False
        )
        
        self.write_idx = 0
        self.read_idx = 0

    def write(self, k: torch.Tensor, v: torch.Tensor) -> int:
        """Write KV pairs to buffer"""
        start_pos = self.write_idx
        seq_len = k.size(0)
        
        curr_block = start_pos // self.block_size
        block_offset = start_pos % self.block_size
        
        for i in range(0, seq_len, self.block_size):
            # Calculate write length for this block
            write_len = min(self.block_size - block_offset, seq_len - i)
            
            # Write to current block
            self.key_buf.data[curr_block, block_offset:block_offset + write_len] = \
                k[i:i + write_len]
            self.value_buf.data[curr_block, block_offset:block_offset + write_len] = \
                v[i:i + write_len]
            self.valid_mask.data[curr_block, block_offset:block_offset + write_len] = True
            
            # Move to next block if needed
            block_offset = (block_offset + write_len) % self.block_size
            if block_offset == 0:
                curr_block = (curr_block + 1) % self.num_blocks
        
        self.write_idx = (start_pos + seq_len) % self.buffer_size
        return start_pos

    def read(self, position: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read valid data up to position"""
        target_block = position // self.block_size
        block_offset = position % self.block_size
        
        if position >= self.buffer_size:
            # Handle wrap-around
            pos_block = (position // self.block_size) % self.num_blocks
            k_valid = torch.cat([
                self.key_buf.data[pos_block+1:],
                self.key_buf.data[:pos_block+1]
            ])
            v_valid = torch.cat([
                self.value_buf.data[pos_block+1:],
                self.value_buf.data[:pos_block+1]
            ])
            mask = torch.cat([
                self.valid_mask.data[pos_block+1:],
                self.valid_mask.data[:pos_block+1]
            ])
        else:
            # Direct read
            k_valid = self.key_buf.data[:target_block + 1].clone()
            v_valid = self.value_buf.data[:target_block + 1].clone()
            mask = self.valid_mask.data[:target_block + 1].clone()
        
        # Mask future positions in last block
        mask[-1, block_offset+1:] = False
        
        return k_valid, v_valid, mask

class TiledAttention:
    """Flash Attention style tiled computation"""
    @staticmethod
    def compute_tile_config(
        seq_len: int,
        num_heads: int,
        block_size: int,
        tile_size: int,
        head_tile_size: int
    ) -> TileConfig:
        """Compute optimal tiling configuration"""
        query_tiles = math.ceil(seq_len / tile_size)
        key_tiles = math.ceil(seq_len / tile_size)
        head_tiles = math.ceil(num_heads / head_tile_size)
        block_tiles = math.ceil(seq_len / block_size)
        
        return TileConfig(
            query_tiles=query_tiles,
            key_tiles=key_tiles,
            head_tiles=head_tiles,
            block_tiles=block_tiles,
            sizes=(tile_size, tile_size, head_tile_size, block_size)
        )

    @staticmethod
    def process_tile(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        scale: float,
        dropout_p: float,
        tile_config: TileConfig
    ) -> torch.Tensor:
        """Process attention for single tile"""
        batch_size = q.size(0)
        q_len = q.size(1)
        
        # Initialize output accumulators
        output = torch.zeros_like(q)
        normalizer = torch.zeros(
            (batch_size, tile_config.head_tiles, q_len, 1),
            device=q.device,
            dtype=q.dtype
        )
        
        # Process tiles
        for h_idx in range(tile_config.head_tiles):
            h_start = h_idx * tile_config.sizes[2]
            h_end = min(h_start + tile_config.sizes[2], q.size(2))
            
            for q_idx in range(tile_config.query_tiles):
                q_start = q_idx * tile_config.sizes[0]
                q_end = min(q_start + tile_config.sizes[0], q_len)
                
                # Initialize accumulator for this query tile
                acc = torch.zeros(
                    (batch_size, h_end - h_start, q_end - q_start, q.size(-1)),
                    device=q.device,
                    dtype=q.dtype
                )
                
                for k_idx in range(tile_config.key_tiles):
                    k_start = k_idx * tile_config.sizes[1]
                    k_end = min(k_start + tile_config.sizes[1], k.size(1))
                    
                    # Get current tiles
                    q_tile = q[:, q_start:q_end, h_start:h_end]
                    k_tile = k[:, k_start:k_end, h_start:h_end]
                    v_tile = v[:, k_start:k_end, h_start:h_end]
                    
                    # Compute attention scores
                    scores = torch.einsum(
                        'bqhd,bkhd->bqhk',
                        q_tile * scale,
                        k_tile
                    )
                    
                    # Apply mask if provided
                    if mask is not None:
                        mask_tile = mask[
                            :,
                            h_start:h_end,
                            q_start:q_end,
                            k_start:k_end
                        ]
                        scores = scores.masked_fill(~mask_tile, float('-inf'))
                    
                    # Apply attention
                    attn_weights = torch.softmax(scores, dim=-1)
                    if dropout_p > 0 and training:
                        attn_weights = torch.nn.functional.dropout(
                            attn_weights,
                            p=dropout_p
                        )
                    
                    # Update accumulator
                    acc += torch.einsum(
                        'bqhk,bkhd->bqhd',
                        attn_weights,
                        v_tile
                    )
                
                # Store results
                output[:, q_start:q_end, h_start:h_end] = acc
        
        return output

@register_attention("ring_buffer", RingBufferConfig)
class RingBufferAttention(Attention):
    """Complete attention implementation with all features"""
    def __init__(self, config: RingBufferConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(config.head_dim)
        
        # Forward and backward KV caches
        self.kv_forward = RingBuffer(
            config.buffer_size,
            config.num_heads,
            config.head_dim,
            config.block_size
        )
        self.kv_backward = RingBuffer(
            config.buffer_size,
            config.num_heads,
            config.head_dim,
            config.block_size
        )
        
        # Decoder feedback buffer
        self.decoder_feedback = RingBuffer(
            config.buffer_size,
            config.num_heads,
            config.head_dim,
            config.block_size
        )
        
        # Tiling configuration
        self.tile_size = config.tile_size
        self.head_tile_size = config.head_tile_size
        self.block_size = config.block_size
        
        # Skip calculation tracking
        self.prefill_size = config.prefill_size
        self.register_buffer(
            'last_calc_pos',
            torch.zeros(1, dtype=torch.long)
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position: Optional[int] = None,
        is_decoder: bool = False,
        att_mask: Optional[torch.Tensor] = None,
        requires_grad: bool = True,
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with all features"""
        if position is None:
            # Encoder phase - store in appropriate buffers
            fwd_pos = self.kv_forward.write(k, v)
            bwd_pos = self.kv_backward.write(k, v) if requires_grad else None
            dec_pos = self.decoder_feedback.write(k, v) if is_decoder else None
            
            return q, BufferState(fwd_pos, bwd_pos, dec_pos, self.last_calc_pos.item())
        
        # Get appropriate KV cache
        kv_cache = self.kv_backward if requires_grad else self.kv_forward
        
        # Get tile configuration
        tile_config = TiledAttention.compute_tile_config(
            position + 1,
            self.num_heads,
            self.block_size,
            self.tile_size,
            self.head_tile_size
        )
        
        outputs = []
        
        # Process encoder attention
        if is_decoder:
            k_enc, v_enc, enc_mask = kv_cache.read(position)
            enc_output = TiledAttention.process_tile(
                q, k_enc, v_enc,
                mask=att_mask,
                scale=self.scale,
                dropout_p=self.dropout if self.training else 0.0,
                tile_config=tile_config
            )
            outputs.append(enc_output)
            
            # Store decoder feedback
            if k is not None and v is not None:
                self.decoder_feedback.write(k, v)
        
        # Process decoder self-attention with skip calculation
        if position > 0:
            # Check if we can skip calculation
            if position - self.last_calc_pos.item() <= self.prefill_size:
                k_dec, v_dec, dec_mask = self.decoder_feedback.read(position)
                dec_output = TiledAttention.process_tile(
                    q, k_dec, v_dec,
                    mask=None,  # Causal mask handled in tiling
                    scale=self.scale,
                    dropout_p=self.dropout if self.training else 0.0,
                    tile_config=tile_config
                )
                outputs.append(dec_output)
            
            self.last_calc_pos = position
        
        # Combine outputs if needed
        if len(outputs) > 1:
            return torch.mean(torch.stack(outputs), dim=0)
        return outputs[0] if outputs else q

    def _use_memory_efficient(self, *tensors) -> bool:
        """Check if we can use xformers memory efficient attention"""
        try:
            return all(t.is_cuda for t in tensors)
        except:
            return False