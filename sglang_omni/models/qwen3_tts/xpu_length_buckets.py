# SPDX-License-Identifier: Apache-2.0
"""Fixed input shapes for the Qwen3-TTS codec decoder on Intel XPU."""

from __future__ import annotations

import logging
import time

import torch

from sglang_omni.models.qwen3_tts.incremental_codec import (
    Qwen3TTSIncrementalCodecState,
    Qwen3TTSIncrementalDecoder,
)

logger = logging.getLogger(__name__)

# note (jagdish): oneDNN builds a kernel per conv input shape, about 3 s for the decoder's 37 convs on an Arc Pro B60, so a frame count it has not seen stalls the request.
BUCKET_FRAMES = 25
# The longest chunk chunked_decode hands the decoder: its 300-frame chunk and 25 frames of left context.
MAX_CHUNK_FRAMES = 325
STREAM_WINDOW_FRAMES = (64, 32, 16, 8, 4, 2, 1)
NUM_QUANTIZERS = 16


def install_length_buckets(decoder: torch.nn.Module) -> None:
    """Pad the decoder's codes on the right to whole buckets and trim the waveform;
    the decoder is causal, so the kept samples do not change."""
    exact_forward = decoder.forward
    samples_per_frame = int(decoder.total_upsample)

    def bucketed_forward(codes: torch.Tensor) -> torch.Tensor:
        frames = int(codes.shape[-1])
        padded_frames = -(-frames // BUCKET_FRAMES) * BUCKET_FRAMES
        if codes.shape[0] > 1:
            # note (jagdish): a batch is a new shape per batch size, so its rows run one at a time on the warmed batch-1 shapes.
            return torch.cat([bucketed_forward(row) for row in codes.split(1)], dim=0)
        elif padded_frames == frames:
            return exact_forward(codes)
        else:
            padded_codes = torch.nn.functional.pad(codes, (0, padded_frames - frames))
            return exact_forward(padded_codes)[..., : frames * samples_per_frame]

    decoder.forward = bucketed_forward


def install_stream_windows(incremental_decoder: Qwen3TTSIncrementalDecoder) -> None:
    """Decode a stream's fresh frames in power-of-two windows; its carried state makes
    the windows' audio the same as one decode of all the frames."""
    exact_decode = incremental_decoder.decode

    def windowed_decode(
        codes: torch.Tensor,
        state: Qwen3TTSIncrementalCodecState,
        *,
        compiled: bool = False,
    ) -> torch.Tensor:
        fresh_frames = int(codes.shape[-1])
        if fresh_frames in STREAM_WINDOW_FRAMES:
            return exact_decode(codes, state, compiled=compiled)
        else:
            pass
        waveform_pieces = []
        frame_offset = 0
        for window_frames in STREAM_WINDOW_FRAMES:
            while fresh_frames - frame_offset >= window_frames:
                window_codes = codes[..., frame_offset : frame_offset + window_frames]
                waveform_pieces.append(
                    exact_decode(window_codes, state, compiled=compiled)
                )
                frame_offset += window_frames
        return torch.cat(waveform_pieces, dim=-1)

    incremental_decoder.decode = windowed_decode


def warm_decoder_shapes(
    decoder: torch.nn.Module,
    incremental_decoder: Qwen3TTSIncrementalDecoder | None,
    device: torch.device,
) -> None:
    """Decode every bucketed chunk width and every stream window once, so no request
    pays for building their kernels."""
    start = time.perf_counter()
    bucket_widths = range(
        BUCKET_FRAMES, MAX_CHUNK_FRAMES + BUCKET_FRAMES, BUCKET_FRAMES
    )
    with torch.inference_mode():
        for frames in bucket_widths:
            decoder(
                torch.zeros(
                    (1, NUM_QUANTIZERS, frames), dtype=torch.long, device=device
                )
            )
        if incremental_decoder is not None:
            for window_frames in STREAM_WINDOW_FRAMES:
                window_codes = torch.zeros(
                    (1, NUM_QUANTIZERS, window_frames), dtype=torch.long, device=device
                )
                state = Qwen3TTSIncrementalCodecState()
                # A fresh state and a carried one take different shapes.
                incremental_decoder.decode(window_codes, state)
                incremental_decoder.decode(window_codes, state)
        else:
            pass
    torch.xpu.synchronize(device)
    logger.info(
        f"Qwen3-TTS vocoder XPU shapes warmed: {len(bucket_widths)} chunk widths in "
        f"{BUCKET_FRAMES}-frame buckets and stream windows {STREAM_WINDOW_FRAMES} "
        f"in {time.perf_counter() - start:.1f} s"
    )
