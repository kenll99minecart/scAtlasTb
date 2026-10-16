"""Embedding mixins vendored from the scvix branch of scvi-tools.

The upstream mixins lack variational batch embeddings, which SCVIX relies on.
Placeholder until SCVIX ships with scvi-tools; remove this module together
with the rest of this package once that happens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.distributions import Normal
from torch.nn import ModuleDict

from scvi import REGISTRY_KEYS
from scvi.module.base import auto_move_data
from scvi.nn import Embedding

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData


class EmbeddingModuleMixin:
    """Mixin class for initializing and using embeddings in a module."""

    @property
    def embeddings_dict(self) -> ModuleDict:
        """Dictionary of embeddings."""
        if not hasattr(self, "_embeddings_dict"):
            self._embeddings_dict = ModuleDict()
        return self._embeddings_dict

    @property
    def embeddings_dim(self) -> dict:
        """Dictionary of embeddings dimensions."""
        if not hasattr(self, "_embeddings_dim"):
            self._embeddings_dim = {}
        return self._embeddings_dim

    @property
    def variational(self) -> dict:
        """Dictionary of whether embedding is variational."""
        if not hasattr(self, "_variational"):
            self._variational = {}
        return self._variational

    def add_embedding(self, key: str, embedding: Embedding, overwrite: bool = False) -> None:
        """Add an embedding to the module."""
        if key in self.embeddings_dict and not overwrite:
            raise KeyError(f"Embedding {key} already exists.")
        torch.nn.init.zeros_(embedding.weight)
        self.embeddings_dict[key] = embedding

    def remove_embedding(self, key: str) -> None:
        """Remove an embedding from the module."""
        if key not in self.embeddings_dict:
            raise KeyError(f"Embedding {key} not found.")
        del self.embeddings_dict[key]

    def get_embedding(
        self,
        key: str,
    ) -> Embedding:
        """Get an embedding from the module."""
        if key not in self.embeddings_dict:
            raise KeyError(f"Embedding {key} not found.")
        return self.embeddings_dict[key]

    def get_embedding_dim(self, key: str, default_value: str | None = None) -> int:
        """Get the dimension of an embedding."""
        if key not in self.embeddings_dim:
            if default_value is not None:
                return default_value
            else:
                raise KeyError(f"Embedding {key} not found.")
        return self.embeddings_dim[key]

    def get_embedding_variational(self, key: str, default_value: str | None = None) -> bool:
        """Get whether an embedding is variational."""
        if key not in self.variational:
            if default_value is not None:
                return default_value
            else:
                raise KeyError(f"Embedding {key} not found.")
        return self.variational[key]

    def init_embedding(
        self,
        key: str,
        num_embeddings: int,
        embedding_dim: int = 5,
        variational: bool = False,
        **kwargs,
    ) -> None:
        """Initialize an embedding in the module."""
        self.embeddings_dim[key] = embedding_dim
        self.variational[key] = variational

        if variational:
            embedding_dim *= 2
        self.add_embedding(key, Embedding(num_embeddings, embedding_dim, **kwargs))

    @auto_move_data
    def compute_embedding(
        self,
        key: str,
        indices: torch.Tensor,
        return_mean: bool = False,
        return_dist: bool = False,
    ) -> torch.Tensor:
        """Forward pass for an embedding."""
        indices = indices.flatten() if indices.ndim > 1 else indices
        embedding = self.get_embedding(key)(indices)
        if self.get_embedding_variational(key):
            embedding_dim = self.get_embedding_dim(key)
            if return_mean:
                return embedding[:, :embedding_dim]
            dist = Normal(embedding[:, :embedding_dim], torch.exp(embedding[:, embedding_dim:]))
            if return_dist:
                return dist
            else:
                return dist.rsample()
        else:
            return embedding


class EmbeddingMixin:
    """Mixin class for initializing and using embeddings in a model.

    Must be used with a module that inherits from :class:`EmbeddingModuleMixin`.
    """

    @torch.inference_mode()
    def get_batch_representation(
        self,
        adata: AnnData | None = None,
        indices: list[int] | None = None,
        batch_size: int | None = None,
        key: str = REGISTRY_KEYS.BATCH_KEY,
        return_mean: bool = True,
    ) -> np.ndarray:
        """Get the batch representation for a given set of indices.

        Parameters
        ----------
        adata
            AnnData object to use.
        indices
            Indices to get the batch representation for.
        batch_size
            Minibatch size for computing the batch representation.
        key
            Setup key to compute the batch representation for.
        return_mean
            Return the mean of the batch representation. Or sample from it.
        """
        if not isinstance(self.module, EmbeddingModuleMixin):
            raise ValueError("The current `module` must inherit from `EmbeddingModuleMixin`.")
        if key not in self.module.embeddings_dim:
            raise ValueError(f"Embedding {key} not found. Enable it during model setup.")

        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        tensors = [
            self.module.compute_embedding(key, tensors[key], return_mean=return_mean)
            for tensors in dataloader
        ]
        return torch.cat(tensors).detach().cpu().numpy()
