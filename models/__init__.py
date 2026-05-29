#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All rights reserved.
#
import warnings

# Suppress "Overwriting fastvit_* in registry" warnings that occur when a
# newer timm version already ships built-in FastViT variants.
# Our local implementation intentionally takes precedence (Apple original,
# with structural-reparameterisation & fork_feat support).
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Overwriting fastvit",
        category=UserWarning,
    )
    from .fastvit import (
        fastvit_t8,
        fastvit_t12,
        fastvit_s12,
        fastvit_sa12,
        fastvit_sa24,
        fastvit_sa36,
        fastvit_ma36,
    )
