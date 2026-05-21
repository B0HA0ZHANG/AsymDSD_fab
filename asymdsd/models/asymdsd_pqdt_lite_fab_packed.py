from __future__ import annotations

from typing import Any

import torch

from .asymdsd_pqdt_fab_packed import PQDTPackedFusedAttnBlockAsymDSD


class PQDTLitePackedFusedAttnBlockAsymDSD(PQDTPackedFusedAttnBlockAsymDSD):
    """PQDT packed FAB variant that trains one randomly selected mask path."""

    _LITE_PATH_NAMES = (
        "sparse_visible",
        "sparse_masked",
        "geometric_halfspace",
        "random",
    )

    def setup(self, stage: str | None = None):
        super().setup(stage)
        self.pqdt_lite_candidate_multi_mask = self.mask_generator.multi_mask
        if self.pqdt_lite_candidate_multi_mask != 4:
            raise ValueError(
                "PQDT lite packed FAB expects mask_generator.multi_mask=4 to "
                "define the four candidate mask paths."
            )
        self.multi_mask = 1

    @torch.no_grad()
    def _generate_packed_masks(
        self,
        masked_attn_weights: list[torch.Tensor],
        masked_centers: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        P = masked_centers.shape[1]
        num_masks_sparse_visible = round(self.sparse_visible_mask_ratio * P)
        num_masks_sparse_masked = round(self.sparse_masked_mask_ratio * P)
        num_masks_geo_halfspace = round(self.geometric_halfspace_mask_ratio * P)
        num_masks_random = round(self.random_mask_ratio * P)

        num_heads = self.student.point_encoder.encoder.config.num_heads
        step = self.global_step
        head_a = step % num_heads
        head_b = (step + num_heads // 2) % num_heads

        path_idx = int(torch.randint(len(self._LITE_PATH_NAMES), ()).item())
        path_name = self._LITE_PATH_NAMES[path_idx]

        if path_name == "sparse_visible":
            selected_mask = self._generate_sparse_only_mask(
                masked_attn_weights,
                num_masks_sparse_visible,
                select_visible=True,
                head_index=head_a,
            )
            selected_head = head_a
        elif path_name == "sparse_masked":
            selected_mask = self._generate_sparse_only_mask(
                masked_attn_weights,
                num_masks_sparse_masked,
                select_visible=False,
                head_index=head_b,
            )
            selected_head = head_b
        elif path_name == "geometric_halfspace":
            selected_mask = self._generate_halfspace_mask(
                masked_centers,
                num_masks_geo_halfspace,
            )
            selected_head = head_a
        elif path_name == "random":
            selected_mask = self._generate_random_patch_mask(
                masked_centers,
                num_masks_random,
            )
            selected_head = head_a
        else:
            raise RuntimeError(f"Unknown PQDT lite mask path: {path_name}")

        all_masks = [selected_mask]
        mask_components = {
            "cls_to_patch": self._compute_cls_to_patch_attention(
                masked_attn_weights,
                head_index=head_a,
            ),
            "cls_to_patch_b": self._compute_cls_to_patch_attention(
                masked_attn_weights,
                head_index=head_b,
            ),
            "block_mask": torch.zeros_like(selected_mask),
            "sparse_mask": ~selected_mask,
            "select_visible": path_name == "sparse_visible",
            "head_a": head_a,
            "head_b": head_b,
            "selected_head": selected_head,
            "selected_path_index": path_idx,
            "selected_path_name": path_name,
            "path_masks": all_masks,
            "path_names": [path_name],
        }
        return all_masks, mask_components
