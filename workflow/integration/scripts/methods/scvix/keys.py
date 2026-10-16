from typing import NamedTuple


class _ScvixKeys(NamedTuple):
    """Registry and module keys introduced by the scvix branch of scvi-tools.

    Placeholder until SCVIX ships with scvi-tools upstream, where these keys
    become part of ``scvi.REGISTRY_KEYS`` and ``scvi.module.base.MODULE_KEYS``.
    """

    ASSAY_KEY: str = "assay"
    ADVERSARIAL_GROUP_KEY: str = "adversarial_group"
    ASSAY_INDEX_KEY: str = "assay_index"
    KL_SAMPLE_KEY: str = "kl_divergence_sample"


SCVIX_KEYS = _ScvixKeys()
