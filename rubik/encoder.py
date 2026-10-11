"""PyTorch Transformer Encoder for Rubik's cube states."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import warnings

# Suppress PyTorch's internal warning when running without optional numpy bridge
warnings.filterwarnings("ignore", message=".*Failed to initialize NumPy.*")

import torch
import torch.nn as nn

# Color vocabulary mapping for the 54 facelet positions
COLOR_TO_ID: Dict[str, int] = {
    "f": 0,
    "u": 1,
    "r": 2,
    "d": 3,
    "l": 4,
    "b": 5,
}
ID_TO_COLOR: Dict[int, str] = {v: k for k, v in COLOR_TO_ID.items()}
CLS_TOKEN_ID: int = 6
VOCAB_SIZE: int = 7
NUM_FACELETS: int = 54


def _state_to_ids(state: Union[str, Sequence[Any]]) -> List[int]:
    """Convert a single 54-element state (tuple or string) into a list of integer token IDs."""
    if len(state) != NUM_FACELETS:
        raise ValueError(f"State must contain exactly 54 facelets, got {len(state)}")
    ids: List[int] = []
    for item in state:
        char = item.lower() if isinstance(item, str) else str(item).lower()
        if char not in COLOR_TO_ID:
            raise ValueError(f"Unknown facelet color '{item}'. Expected one of {list(COLOR_TO_ID.keys())}")
        ids.append(COLOR_TO_ID[char])
    return ids


def cube_to_tensor(
    state: Union[str, Sequence[Any]], device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert a single cube state into a (1, 54) LongTensor.

    Args:
        state: 54-element sequence or 54-character string.
        device: Target torch device.

    Returns:
        torch.LongTensor of shape (1, 54).
    """
    ids = _state_to_ids(state)
    return torch.tensor([ids], dtype=torch.long, device=device)


def batch_to_tensor(
    states: Sequence[Union[str, Sequence[Any]]], device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert a list/batch of cube states into a (B, 54) LongTensor.

    Args:
        states: Sequence of 54-element states.
        device: Target torch device.

    Returns:
        torch.LongTensor of shape (B, 54).
    """
    batch_ids = [_state_to_ids(s) for s in states]
    return torch.tensor(batch_ids, dtype=torch.long, device=device)


class CubeTransformerEncoder(nn.Module):
    """
    Transformer Encoder for Rubik's cube representation learning.

    Converts 54-facelet cube sequences into dense vector representations using
    spatial positional embeddings and multi-head self-attention.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        d_embed: Optional[int] = None,
        pooling: str = "cls",
        activation: str = "gelu",
    ):
        """
        Args:
            d_model: Hidden dimensionality of transformer layers.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Hidden dimension of feedforward networks.
            dropout: Dropout probability.
            d_embed: Output embedding dimension (defaults to d_model).
            pooling: Global pooling method ('cls' or 'mean').
            activation: Activation function ('gelu' or 'relu').
        """
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError(f"Unsupported pooling mode '{pooling}'. Must be 'cls' or 'mean'.")

        self.d_model = d_model
        self.d_embed = d_embed if d_embed is not None else d_model
        self.pooling = pooling

        # Token embedding (6 colors + optional CLS token)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)

        # Spatial positional embeddings
        num_positions = NUM_FACELETS + 1 if pooling == "cls" else NUM_FACELETS
        self.pos_embedding = nn.Embedding(num_positions, d_model)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.input_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Output projection head
        if self.d_embed != d_model:
            self.proj = nn.Linear(d_model, self.d_embed)
        else:
            self.proj = nn.Identity()

        self.output_norm = nn.LayerNorm(self.d_embed)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass converting cube token tensor into embeddings.

        Args:
            x: torch.LongTensor of shape (B, 54).

        Returns:
            Tuple of:
                - global_embedding: torch.Tensor of shape (B, d_embed)
                - token_embeddings: torch.Tensor of shape (B, 54, d_model)
        """
        batch_size, seq_len = x.shape
        if seq_len != NUM_FACELETS:
            raise ValueError(f"Input sequence length must be 54, got {seq_len}")

        tokens = self.token_embedding(x)  # (B, 54, d_model)

        if self.pooling == "cls":
            cls = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
            h = torch.cat([cls, tokens], dim=1)  # (B, 55, d_model)
            positions = torch.arange(55, device=x.device).unsqueeze(0)  # (1, 55)
            h = h + self.pos_embedding(positions)
            h = self.dropout(self.input_norm(h))

            out = self.transformer(h)  # (B, 55, d_model)
            global_rep = out[:, 0, :]  # (B, d_model)
            token_reps = out[:, 1:, :]  # (B, 54, d_model)
        else:
            positions = torch.arange(54, device=x.device).unsqueeze(0)  # (1, 54)
            h = tokens + self.pos_embedding(positions)
            h = self.dropout(self.input_norm(h))

            out = self.transformer(h)  # (B, 54, d_model)
            global_rep = out.mean(dim=1)  # (B, d_model)
            token_reps = out  # (B, 54, d_model)

        global_embedding = self.output_norm(self.proj(global_rep))
        return global_embedding, token_reps

    @torch.no_grad()
    def embed(
        self, states: Union[str, Sequence[Any], Sequence[Union[str, Sequence[Any]]]]
    ) -> torch.Tensor:
        """
        Convenience method to encode one or more states directly.

        Args:
            states: A single state (54-char string / tuple) or sequence of states.

        Returns:
            torch.Tensor of shape (B, d_embed) or (1, d_embed).
        """
        self.eval()
        device = next(self.parameters()).device
        if isinstance(states, str) or (isinstance(states, (tuple, list)) and len(states) == 54 and isinstance(states[0], str) and len(states[0]) == 1):
            tensor = cube_to_tensor(states, device=device)
        else:
            tensor = batch_to_tensor(states, device=device)

        global_emb, _ = self.forward(tensor)
        return global_emb
