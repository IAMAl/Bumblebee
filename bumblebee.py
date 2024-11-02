import torch
import torch.nn as nn
from typing import Optional, Tuple, NamedTuple, Union
from dataclasses import dataclass
from xformers.components import Attention, AttentionConfig
from xformers.components.attention import register_attention
from enum import Enum, auto
import math

class TileType(Enum):
    """Types of attention computation"""
    DENSE = "dense"     # Full computation
    TRIANGLE = "triangle" # Special handling for diagonal tiles
    SKIP = "skip"      # Skip computation (outside causal mask)

class TileOrder(Enum):
    """Tile processing order strategies"""
    Q_MAJOR = auto()  # Optimize for query reuse
    K_MAJOR = auto()  # Optimize for key/value reuse

@dataclass
class RingBufferConfig(AttentionConfig):
    """xFormers compatible configuration for ring buffer attention"""
    # Required xFormers fields
    num_heads: int
    head_dim: int
    dropout: float = 0.0
    attention_dropout: Optional[float] = None
    seq_len: Optional[int] = None

    # Ring buffer specific configuration
    buffer_size: int = 2048
    block_size: int = 128
    tile_size: int = 32
    head_tile_size: int = 4
    prefill_size: int = 512
    causal: bool = True

class BufferState(NamedTuple):
    """Track states of buffers and positions"""
    kv_fwd_pos: int
    kv_bwd_pos: Optional[int]
    dec_pos: Optional[int]
    last_calc: int

@dataclass
class TileInfo:
    """Information for processing a specific tile"""
    type: TileType     # Type of computation needed
    q_start: int       # Start of query sequence
    q_end: int         # End of query sequence
    k_start: int       # Start of key sequence
    k_end: int         # End of key sequence
    h_start: int       # Start of head dimension
    h_end: int         # End of head dimension

class RingBuffer(nn.Module):
    """Ring buffer implementation"""
    def __init__(self,
                buffer_size: int,
                num_heads: int,
                head_dim: int,
                block_size: int):
        super().__init__()
        self.buffer_size = buffer_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.num_blocks = (buffer_size + block_size - 1) // block_size

        # Register buffers properly for state_dict handling
        self.register_buffer(
            'key_buf',
            torch.zeros(self.num_blocks, block_size, num_heads, head_dim)
        )
        self.register_buffer(
            'value_buf',
            torch.zeros(self.num_blocks, block_size, num_heads, head_dim)
        )
        self.register_buffer(
            'valid_mask',
            torch.zeros(self.num_blocks, block_size, dtype=torch.bool)
        )

        self.reset_buffers()

    def reset_buffers(self):
        """Reset buffer states"""
        self.write_idx = 0
        self.read_idx = 0
        self.valid_mask.fill_(False)

    def write(self, k: torch.Tensor, v: torch.Tensor) -> int:
        """Write KV pairs to buffer"""
        start_pos = self.write_idx
        seq_len = k.size(0)

        curr_block = start_pos // self.block_size
        block_offset = start_pos % self.block_size

        remaining = seq_len
        curr_pos = 0

        while remaining > 0:
            write_len = min(self.block_size - block_offset, remaining)

            self.key_buf[curr_block, block_offset:block_offset + write_len] = \
                k[curr_pos:curr_pos + write_len]
            self.value_buf[curr_block, block_offset:block_offset + write_len] = \
                v[curr_pos:curr_pos + write_len]
            self.valid_mask[curr_block, block_offset:block_offset + write_len] = True

            remaining -= write_len
            curr_pos += write_len
            block_offset = (block_offset + write_len) % self.block_size
            if block_offset == 0:
                curr_block = (curr_block + 1) % self.num_blocks

        self.write_idx = (start_pos + seq_len) % self.buffer_size
        return start_pos

    def read(self, position: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read valid data up to position"""
        if position >= self.buffer_size:
            position = position % self.buffer_size

        target_block = position // self.block_size
        block_offset = position % self.block_size

        # Get valid blocks
        if self.write_idx <= position:
            valid_blocks = slice(0, target_block + 1)
        else:
            valid_blocks = torch.cat([
                torch.arange(self.write_idx // self.block_size, self.num_blocks),
                torch.arange((position // self.block_size) + 1)
            ])

        k_valid = self.key_buf[valid_blocks]
        v_valid = self.value_buf[valid_blocks]
        mask = self.valid_mask[valid_blocks]

        if mask.size(0) > 0:
            mask[-1, block_offset+1:] = False

        return k_valid, v_valid, mask

class TileOrder(Enum):
    """Tile processing order strategies"""
    Q_MAJOR = auto()  # Optimize for query reuse
    K_MAJOR = auto()  # Optimize for key/value reuse

class HierarchicalTilingProcessor:
    """Two-level hierarchical tile processing system"""
    def __init__(
        self,
        tile_size: int,
        head_tile_size: int,
        causal: bool = True,
        tile_order: TileOrder = TileOrder.Q_MAJOR
    ):
        self.tile_size = tile_size
        self.head_tile_size = head_tile_size
        self.causal = causal
        self.tile_order = tile_order

    def get_tile_type(
        self,
        q_idx: int,
        k_idx: int,
        causal: bool = True
    ) -> TileType:
        """Determine tile computation type based on position"""
        if not causal:
            return TileType.DENSE

        if k_idx > q_idx:
            return TileType.SKIP  # Future tokens
        elif k_idx == q_idx:
            return TileType.TRIANGLE  # Diagonal tile
        else:
            return TileType.DENSE  # Past tokens

    def process_dense_tile(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tile: TileInfo,
        scale: float,
        acc_output: torch.Tensor,
        acc_normalizer: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        training: bool = True
    ) -> None:
        """Process a dense tile with full attention computation"""
        # Extract tile tensors
        q_tile = q[:, tile.q_start:tile.q_end, tile.h_start:tile.h_end]
        k_tile = k[:, tile.k_start:tile.k_end, tile.h_start:tile.h_end]
        v_tile = v[:, tile.k_start:tile.k_end, tile.h_start:tile.h_end]

        # Compute attention scores
        scores = torch.einsum('bqhd,bkhd->bhqk', q_tile * scale, k_tile)

        # Apply attention mask if provided
        if att_mask is not None:
            mask_tile = att_mask[
                :,
                tile.h_start:tile.h_end,
                tile.q_start:tile.q_end,
                tile.k_start:tile.k_end
            ]
            scores.masked_fill_(~mask_tile, float('-inf'))

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            pad_mask = key_padding_mask[:, tile.k_start:tile.k_end]
            scores.masked_fill_(
                pad_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        if dropout_p > 0.0 and training:
            attn_weights = torch.nn.functional.dropout(
                attn_weights,
                p=dropout_p
            )

        # Accumulate results
        acc_normalizer[:, :, tile.q_start:tile.q_end] += attn_weights.sum(dim=-1, keepdim=True)
        acc_output[:, tile.q_start:tile.q_end, tile.h_start:tile.h_end] += torch.einsum(
            'bhqk,bkhd->bqhd', attn_weights, v_tile
        )

    def process_triangle_tile(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tile: TileInfo,
        scale: float,
        acc_output: torch.Tensor,
        acc_normalizer: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        training: bool = True
    ) -> None:
        """Process a diagonal tile with triangular attention pattern"""
        # Extract tile tensors
        q_tile = q[:, tile.q_start:tile.q_end, tile.h_start:tile.h_end]
        k_tile = k[:, tile.k_start:tile.k_end, tile.h_start:tile.h_end]
        v_tile = v[:, tile.k_start:tile.k_end, tile.h_start:tile.h_end]

        # Compute attention scores
        scores = torch.einsum('bqhd,bkhd->bhqk', q_tile * scale, k_tile)

        # Create triangular mask
        seq_len = tile.q_end - tile.q_start
        triangle_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device),
            diagonal=1
        )
        scores.masked_fill_(triangle_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply additional masks
        if att_mask is not None:
            mask_tile = att_mask[
                :,
                tile.h_start:tile.h_end,
                tile.q_start:tile.q_end,
                tile.k_start:tile.k_end
            ]
            scores.masked_fill_(~mask_tile, float('-inf'))

        if key_padding_mask is not None:
            pad_mask = key_padding_mask[:, tile.k_start:tile.k_end]
            scores.masked_fill_(
                pad_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        if dropout_p > 0.0 and training:
            attn_weights = torch.nn.functional.dropout(
                attn_weights,
                p=dropout_p
            )

        # Accumulate results
        acc_normalizer[:, :, tile.q_start:tile.q_end] += attn_weights.sum(dim=-1, keepdim=True)
        acc_output[:, tile.q_start:tile.q_end, tile.h_start:tile.h_end] += torch.einsum(
            'bhqk,bkhd->bqhd', attn_weights, v_tile
        )

    def process_hierarchical(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        att_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        training: bool = True
    ) -> torch.Tensor:
        """Process attention using hierarchical tiling with optimized ordering"""
        batch_size, seq_len, num_heads, head_dim = q.shape

        # Initialize accumulators
        output = torch.zeros_like(q)

        # Calculate number of tiles
        q_tiles = math.ceil(seq_len / self.tile_size)
        k_tiles = math.ceil(seq_len / self.tile_size)
        h_tiles = math.ceil(num_heads / self.head_tile_size)

        if self.tile_order == TileOrder.Q_MAJOR:
            # Q-major ordering: Optimize for query reuse
            # Process order: heads -> query -> key
            # This maximizes reuse of query tiles across different key tiles
            for h_idx in range(h_tiles):
                h_start = h_idx * self.head_tile_size
                h_end = min(h_start + self.head_tile_size, num_heads)

                for q_idx in range(q_tiles):
                    # Initialize normalizer for this query tile
                    normalizer = torch.zeros(
                        (batch_size, num_heads, self.tile_size, 1),
                        device=q.device,
                        dtype=q.dtype
                    )

                    q_start = q_idx * self.tile_size
                    q_end = min(q_start + self.tile_size, seq_len)

                    # Load query tile once and reuse
                    q_tile = q[:, q_start:q_end, h_start:h_end]

                    # Process all relevant key tiles for this query tile
                    max_k_tiles = q_tiles if not self.causal else (q_idx + 1)
                    for k_idx in range(max_k_tiles):
                        k_start = k_idx * self.tile_size
                        k_end = min(k_start + self.tile_size, seq_len)

                        tile_type = self.get_tile_type(q_idx, k_idx, self.causal)
                        if tile_type == TileType.SKIP:
                            continue

                        tile = TileInfo(
                            type=tile_type,
                            q_start=q_start,
                            q_end=q_end,
                            k_start=k_start,
                            k_end=k_end,
                            h_start=h_start,
                            h_end=h_end
                        )

                        if tile_type == TileType.DENSE:
                            self.process_dense_tile(
                                q_tile, k, v,  # Pass pre-loaded q_tile
                                tile=tile,
                                scale=scale,
                                acc_output=output,
                                acc_normalizer=normalizer,
                                att_mask=att_mask,
                                key_padding_mask=key_padding_mask,
                                dropout_p=dropout_p,
                                training=training
                            )
                        else:  # TRIANGLE
                            self.process_triangle_tile(
                                q_tile, k, v,  # Pass pre-loaded q_tile
                                tile=tile,
                                scale=scale,
                                acc_output=output,
                                acc_normalizer=normalizer,
                                att_mask=att_mask,
                                key_padding_mask=key_padding_mask,
                                dropout_p=dropout_p,
                                training=training
                            )

                    # Normalize the output for this query tile
                    output[:, q_start:q_end, h_start:h_end] /= (normalizer + 1e-6)

        else:  # TileOrder.K_MAJOR
            # K-major ordering: Optimize for key/value reuse
            # Process order: heads -> key -> query
            # This maximizes reuse of key/value tiles across different query tiles
            for h_idx in range(h_tiles):
                h_start = h_idx * self.head_tile_size
                h_end = min(h_start + self.head_tile_size, num_heads)

                for k_idx in range(k_tiles):
                    k_start = k_idx * self.tile_size
                    k_end = min(k_start + self.tile_size, seq_len)

                    # Load key/value tiles once and reuse
                    k_tile = k[:, k_start:k_end, h_start:h_end]
                    v_tile = v[:, k_start:k_end, h_start:h_end]

                    # Process all query tiles that can use this key tile
                    min_q_tiles = k_idx if self.causal else 0
                    for q_idx in range(min_q_tiles, q_tiles):
                        # Initialize normalizer for this query tile
                        normalizer = torch.zeros(
                            (batch_size, num_heads, self.tile_size, 1),
                            device=q.device,
                            dtype=q.dtype
                        )

                        q_start = q_idx * self.tile_size
                        q_end = min(q_start + self.tile_size, seq_len)

                        tile_type = self.get_tile_type(q_idx, k_idx, self.causal)
                        if tile_type == TileType.SKIP:
                            continue

                        tile = TileInfo(
                            type=tile_type,
                            q_start=q_start,
                            q_end=q_end,
                            k_start=k_start,
                            k_end=k_end,
                            h_start=h_start,
                            h_end=h_end
                        )

                        if tile_type == TileType.DENSE:
                            self.process_dense_tile(
                                q, k_tile, v_tile,  # Pass pre-loaded k,v tiles
                                tile=tile,
                                scale=scale,
                                acc_output=output,
                                acc_normalizer=normalizer,
                                att_mask=att_mask,
                                key_padding_mask=key_padding_mask,
                                dropout_p=dropout_p,
                                training=training
                            )
                        else:  # TRIANGLE
                            self.process_triangle_tile(
                                q, k_tile, v_tile,  # Pass pre-loaded k,v tiles
                                tile=tile,
                                scale=scale,
                                acc_output=output,
                                acc_normalizer=normalizer,
                                att_mask=att_mask,
                                key_padding_mask=key_padding_mask,
                                dropout_p=dropout_p,
                                training=training
                            )

                        # Normalize the output for this query tile
                        output[:, q_start:q_end, h_start:h_end] /= (normalizer + 1e-6)

        return output

@register_attention("ring_buffer", RingBufferConfig)
class RingBufferAttention(Attention):
    """xFormers compatible attention implementation with hierarchical tiling"""
    def __init__(self, config: RingBufferConfig):
        super().__init__()
        self.config = config
        self.scale = 1.0 / math.sqrt(config.head_dim)
        self.dropout_p = config.dropout

        # Initialize hierarchical tiling processor
        self.tiling_processor = HierarchicalTilingProcessor(
            tile_size=config.tile_size,
            head_tile_size=config.head_tile_size,
            causal=config.causal
        )

        # Initialize ring buffers
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
        self.decoder_feedback = RingBuffer(
            config.buffer_size,
            config.num_heads,
            config.head_dim,
            config.block_size
        )

        # For tracking
        self.register_buffer('last_calc_pos', torch.zeros(1, dtype=torch.long))
        self._seq_len_cached = 0

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        needs_weights: bool = False,
        output_attentions: bool = False,
        position: Optional[int] = None,
        is_decoder: bool = False,
        requires_grad: bool = True,
        *args,
        **kwargs
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
    ]:
        """xFormers compatible forward pass with hierarchical tiling"""
        # Validate inputs
        self._validate_input(q, k, v)

        # Handle encoder phase
        if position is None:
            fwd_pos = self.kv_forward.write(k, v)
            bwd_pos = self.kv_backward.write(k, v) if requires_grad else None
            dec_pos = self.decoder_feedback.write(k, v) if is_decoder else None
            return q, BufferState(fwd_pos, bwd_pos, dec_pos, self.last_calc_pos.item())

        # Get KV cache
        kv_cache = self.kv_backward if requires_grad else self.kv_forward
        k_cached, v_cached, cache_mask = kv_cache.read(position)

        # Combine masks
        if att_mask is not None and cache_mask is not None:
            att_mask = att_mask & cache_mask
        elif cache_mask is not None:
            att_mask = cache_mask

        # Process attention using hierarchical tiling
        output = self.tiling_processor.process_hierarchical(
            q, k_cached, v_cached,
            scale=self.scale,
            att_mask=att_mask,
            key_padding_mask=key_padding_mask,
            dropout_p=self.dropout_p,
            training=self.training
        )

        # Update decoder feedback if needed
        if is_decoder and k is not None and v is not None:
            self.decoder_feedback.write(k, v)

        self.last_calc_pos = torch.tensor([position], device=q.device)

        if needs_weights or output_attentions:
            # Use xFormers compatible method for attention weights
            context, attn_weights = self.get_attention_scores_and_context(
                q, k_cached, v_cached,
                att_mask,
                self.scale,
                key_padding_mask=key_padding_mask
            )

            if output_attentions:
                return output, attn_weights
            return output, None

        return output

    @staticmethod
    def get_attention_scores_and_context(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor],
        scale: float,
        key_padding_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """xFormers compatible method to compute attention scores and context"""
        batch_size, seq_len, num_heads, head_dim = q.shape

        # Reshape for batch matrix multiplication
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)  # [B, H, S, D]
        v = v.transpose(1, 2)  # [B, H, S, D]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if att_mask is not None:
            scores = scores.masked_fill(~att_mask, float('-inf'))

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        att_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(att_weights, v)

        # Reshape back
        context = context.transpose(1, 2)  # [B, S, H, D]

        return context, att_weights

    def _validate_input(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> None:
        """Validate input tensors"""
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"Query, key, and value must be 4D tensors, "
                f"got q:{q.dim()}D, k:{k.dim()}D, v:{v.dim()}D"
            )

        if not (q.size(-1) == k.size(-1) == v.size(-1) == self.config.head_dim):
            raise ValueError(
                f"Head dimension mismatch. Expected {self.config.head_dim}, "
                f"got q:{q.size(-1)}, k:{k.size(-1)}, v:{v.size(-1)}"
            )

        if not (q.size(-2) == k.size(-2) == v.size(-2) == self.config.num_heads):
            raise ValueError(
                f"Number of heads mismatch. Expected {self.config.num_heads}, "
                f"got q:{q.size(-2)}, k:{k.size(-2)}, v:{v.size(-2)}"
            )