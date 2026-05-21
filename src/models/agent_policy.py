import torch
import torch.nn as nn

from src.models.lstm import PolicyLSTM
from src.models.visual_encoder import DinoV2Encoder


class AgentPolicy(nn.Module):
    """Policy network: visual encoder + aux features → LSTM → actor/critic heads.

    The target-object embedding is *learned* over a closed vocabulary fixed
    at construction time (passed in from the YAML via train.py). This replaces
    the previous SHA256-hash-based embedding, which had no semantic structure
    and could not generalise: two visually similar objects (e.g. Mug vs Cup)
    received uncorrelated random vectors, forcing the policy to memorise each
    target independently.
    """

    def __init__(
        self,
        n_actions: int,
        target_object_vocab: list[str],
        target_object_embed_dim: int = 8,
        encoder_name: str = "dinov2_vitb14",
        hidden_dim: int = 512,
        lstm_layers: int = 1,
        device: str | torch.device | None = None,
    ):
        super().__init__()

        if not target_object_vocab:
            raise ValueError(
                "AgentPolicy requires a non-empty target_object_vocab. "
                "Pass training.target.candidates from the YAML."
            )

        # Vocab + idx mapping. We store the *list* (not just the size) so we
        # can save it in the checkpoint and detect mismatches at resume.
        self.target_object_vocab = list(target_object_vocab)
        self.target_object_embed_dim = target_object_embed_dim
        self.n_actions = n_actions

        # Learnable embedding table. Small init keeps it from dominating the
        # other aux features at the start of training (compass, GPS, etc. are
        # in [-1, 1]; we want target_embedding to start at the same scale).
        self.target_embedding = nn.Embedding(
            num_embeddings=len(self.target_object_vocab),
            embedding_dim=target_object_embed_dim,
        )
        nn.init.normal_(self.target_embedding.weight, mean=0.0, std=0.1)

        self.visual_encoder = DinoV2Encoder(
            model_name=encoder_name,
            device=device,
        )
        vis_dim = self.visual_encoder.output_dim

        # Recurrent + heads. input_dim mirrors the layout produced by
        # ThorEnv.get_aux_features(), *with* the idx already expanded to its
        # embedding (i.e. + target_object_embed_dim, not +1 for the idx).
        gps_dim = 2
        compass_dim = 4
        resolution_dim = 1
        budget_dim = 1
        aux_dim_expanded = (
            gps_dim + compass_dim + n_actions + resolution_dim + budget_dim
            + target_object_embed_dim
        )
        self.lstm_core = PolicyLSTM(
            vis_dim=vis_dim,
            n_actions=n_actions,
            target_object_embed_dim=target_object_embed_dim,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _expand_target_idx(self, aux_features: torch.Tensor) -> torch.Tensor:
        """Replace the last scalar of aux_features (the target-object idx)
        by its learned embedding vector.

        Args:
            aux_features: shape (..., D_aux_raw) where the last element is the
                          target-object idx as a float scalar (cast from int).

        Returns:
            Tensor of shape (..., D_aux_raw - 1 + target_object_embed_dim).
        """
        aux_main = aux_features[..., :-1]
        target_idx = aux_features[..., -1].long()
        # Safety check: out-of-vocab idx would silently wrap in nn.Embedding
        # on some PyTorch versions. We assert explicitly.
        if torch.any(target_idx < 0) or torch.any(
            target_idx >= len(self.target_object_vocab)
        ):
            raise IndexError(
                f"target_idx out of range for vocab of size "
                f"{len(self.target_object_vocab)}: got {target_idx.tolist()}"
            )
        target_vec = self.target_embedding(target_idx)
        return torch.cat([aux_main, target_vec], dim=-1)

    def forward(
        self,
        image: torch.Tensor,
        aux_features: torch.Tensor,
        hidden=None,
    ):
        """
        Args:
            image: (B, 3, H, W) or (3, H, W). Will be batched automatically.
            aux_features: (B, D_aux_raw) or (D_aux_raw,), last element = target idx.
            hidden: LSTM hidden state, or None.
        Returns:
            logits: (B, n_actions)
            value:  (B, 1)
            hidden: new LSTM state
        """
        if aux_features.dim() == 1:
            aux_features = aux_features.unsqueeze(0)
        
        aux_full = self._expand_target_idx(aux_features)
        
        with torch.no_grad():
            visual_features = self.visual_encoder(image)

        obs = torch.cat(
            [visual_features, aux_full],
            dim=-1,
        )

        logits, value, hidden = self.lstm_core(obs, hidden)
        return logits, value, hidden