"""scVI wrapper — used only when real scRNA-seq data is loaded.

For the synthetic toy, the cell state lives directly in R^d and scVI is
bypassed. When `iwm.data.perturb_seq_loader` is wired in, this wrapper
loads a trained scVI model and exposes the latent encoder.

scVI is BSD-3 (commercial-clean). Keeping the import optional keeps the
toy path dependency-free.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ScviEncoder:
    """Thin adapter over scvi.model.SCVI.

    Usage:
        enc = ScviEncoder.from_trained("path/to/model_dir")
        z = enc.encode(adata)
    """

    def __init__(self, model: "object") -> None:
        self._model = model

    @classmethod
    def from_trained(cls, model_dir: str, adata: Optional["object"] = None) -> "ScviEncoder":
        try:
            import scvi  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "scvi-tools is an optional dependency; install with `pip install iwm[real]`."
            ) from e
        if adata is None:
            raise ValueError("scVI requires an AnnData argument to rebind at load.")
        model = scvi.model.SCVI.load(model_dir, adata=adata)
        return cls(model)

    def encode(self, adata: "object") -> np.ndarray:
        return self._model.get_latent_representation(adata)

    def decode(
        self,
        z: np.ndarray,
        adata_ref: "object",
        log1p: bool = True,
        library_size: float = 1e4,
    ) -> np.ndarray:
        """Decode latent vectors to log-normalised gene expression.

        Returns (N, n_genes) float32 in log1p(CPM) space by default.

        NOTE: scVI's VAE decoder is smooth/regularised — LFC magnitudes are
        compressed ~100-1000x vs real single-cell data. Use `knn_decode` for
        interpretable gene-level analysis.
        """
        expr = self._model.get_normalized_expression(
            adata=adata_ref,
            get_from_latent=True,
            latent=z,
            library_size=library_size,
            return_numpy=True,
        )
        if log1p:
            expr = np.log1p(expr)
        return expr.astype(np.float32)

    def knn_decode(
        self,
        z_pred: np.ndarray,
        z_ref: np.ndarray,
        expr_ref: np.ndarray,
        k: int = 10,
    ) -> np.ndarray:
        """Decode predicted latent vectors via k-NN retrieval in latent space.

        Finds the k nearest reference cells for each predicted z, then
        returns their mean log1p-normalised expression. Gives realistic LFC
        magnitudes (~1-3) because it uses actual observed gene expression.

        Parameters
        ----------
        z_pred    : (N, d) predicted latent vectors from IWM
        z_ref     : (M, d) reference latent vectors (all training cells)
        expr_ref  : (M, G) log1p-normalised expression of reference cells
        k         : number of nearest neighbours to average
        """
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        nn.fit(z_ref)
        _, idx = nn.kneighbors(z_pred)           # (N, k)
        return expr_ref[idx].mean(axis=1).astype(np.float32)  # (N, G)

    @property
    def gene_names(self) -> list:
        var = self._model.adata.var
        if "feature_name" in var.columns:
            return list(var["feature_name"])
        return list(self._model.adata.var_names)
