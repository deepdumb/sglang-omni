# SPDX-License-Identifier: Apache-2.0
"""The XPU length buckets give the decoder's own audio from a fixed set of shapes."""

from __future__ import annotations

import torch

from sglang_omni.models.qwen3_tts import xpu_length_buckets as buckets

SAMPLES_PER_FRAME = 4


class CausalDecoder(torch.nn.Module):
    """A causal stand-in for the codec decoder: codes (B, Q, T) to audio (B, 1, T * 4)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(2048, 8, dtype=torch.float64)
        self.conv = torch.nn.Conv1d(
            8, SAMPLES_PER_FRAME, kernel_size=5, dtype=torch.float64
        )
        self.total_upsample = SAMPLES_PER_FRAME
        self.frames_seen: list[tuple[int, int]] = []

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        self.frames_seen.append((int(codes.shape[0]), int(codes.shape[-1])))
        hidden_states = self.embed(codes).sum(dim=1).transpose(1, 2)
        hidden_states = self.conv(torch.nn.functional.pad(hidden_states, (4, 0)))
        return hidden_states.transpose(1, 2).reshape(codes.shape[0], 1, -1)


def test_bucketed_decode_matches_the_exact_length_decode() -> None:
    exact = CausalDecoder()
    bucketed = CausalDecoder()
    buckets.install_length_buckets(bucketed)
    generator = torch.Generator().manual_seed(1)
    for frames in (1, 24, 25, 26, 57, 137):
        codes = torch.randint(0, 2048, (1, 16, frames), generator=generator)
        with torch.no_grad():
            torch.testing.assert_close(bucketed(codes), exact(codes))
    assert all(
        frames % buckets.BUCKET_FRAMES == 0 for _, frames in bucketed.frames_seen
    )


def test_a_batch_decodes_row_by_row_at_bucketed_widths() -> None:
    exact = CausalDecoder()
    bucketed = CausalDecoder()
    buckets.install_length_buckets(bucketed)
    codes = torch.randint(
        0, 2048, (3, 16, 40), generator=torch.Generator().manual_seed(2)
    )
    with torch.no_grad():
        torch.testing.assert_close(bucketed(codes), exact(codes))
    assert bucketed.frames_seen == [(1, 50)] * 3


class SteppedDecoder:
    """A stateful stand-in for the incremental decoder: each step's audio depends on
    every frame decoded before it, through the state."""

    def __init__(self) -> None:
        self.frames_seen: list[int] = []

    def decode(
        self,
        codes: torch.Tensor,
        state: dict[str, torch.Tensor],
        *,
        compiled: bool = False,
    ) -> torch.Tensor:
        self.frames_seen.append(int(codes.shape[-1]))
        running_total = state.setdefault("total", torch.zeros(codes.shape[0]))
        totals_per_frame = []
        for frame_codes in codes.unbind(-1):
            running_total = running_total + frame_codes.float().sum(dim=1)
            totals_per_frame.append(running_total.clone())
        state["total"] = running_total
        return torch.stack(totals_per_frame, dim=-1)


def test_stream_windows_decode_the_same_audio_in_power_of_two_steps() -> None:
    exact = SteppedDecoder()
    windowed = SteppedDecoder()
    buckets.install_stream_windows(windowed)
    exact_state: dict[str, torch.Tensor] = {}
    windowed_state: dict[str, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(3)
    for fresh_frames in (16, 11, 8, 37, 1):
        codes = torch.randint(0, 2048, (2, 16, fresh_frames), generator=generator)
        torch.testing.assert_close(
            windowed.decode(codes, windowed_state),
            exact.decode(codes, exact_state),
            rtol=0,
            atol=0,
        )
    assert set(windowed.frames_seen) <= set(buckets.STREAM_WINDOW_FRAMES)
