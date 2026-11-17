"""Length buckets for the Qwen3-TTS codec decoder on XPU.

oneDNN builds a convolution kernel per input shape, about 3 s for the decoder's 37
convolutions the first time a frame count is seen. The decoder is causal, so padding the
codes on the right to a multiple of BUCKET_FRAMES and trimming the waveform gives the
same audio from a small, warmable set of shapes."""

from __future__ import annotations

import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

BUCKET_FRAMES = int(os.environ.get("SGLANG_OMNI_XPU_VOCODER_BUCKET_FRAMES", 25))
# chunked_decode's chunk and left context: a chunk carries at most this many frames.
MAX_CHUNK_FRAMES = 325
NUM_QUANTIZERS = 16


def install_length_buckets(decoder: torch.nn.Module) -> bool:
    """Wrap the decoder's forward once; returns whether the buckets are in place."""
    if BUCKET_FRAMES <= 1:
        return False
    else:
        pass
    if getattr(decoder, "xpu_bucket_frames", None) is not None:
        return True
    else:
        pass
    forward = decoder.forward
    samples_per_frame = decoder.total_upsample

    def bucketed_forward(codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[0] > 1:
            # A batch is a new shape per batch size; its rows one by one are warm.
            return torch.cat([bucketed_forward(row) for row in codes.split(1)], dim=0)
        else:
            pass
        frames = codes.shape[-1]
        padded = -(-frames // BUCKET_FRAMES) * BUCKET_FRAMES
        if padded == frames:
            return forward(codes)
        else:
            waveform = forward(torch.nn.functional.pad(codes, (0, padded - frames)))
            return waveform[..., : frames * samples_per_frame]

    decoder.forward = bucketed_forward
    decoder.xpu_bucket_frames = BUCKET_FRAMES
    return True


def warm_length_buckets(decoder: torch.nn.Module, device: torch.device) -> None:
    """Decode one batch-1 chunk of every bucketed length so no request pays for a kernel."""
    if getattr(decoder, "xpu_bucket_frames", None) is None or getattr(
        decoder, "xpu_buckets_warm", False
    ):
        return
    else:
        pass
    start = time.perf_counter()
    lengths = range(BUCKET_FRAMES, MAX_CHUNK_FRAMES + BUCKET_FRAMES, BUCKET_FRAMES)
    with torch.inference_mode():
        for frames in lengths:
            codes = torch.zeros(
                (1, NUM_QUANTIZERS, frames), dtype=torch.long, device=device
            )
            decoder(codes)
    torch.xpu.synchronize(device)
    decoder.xpu_buckets_warm = True
    logger.info(
        "Qwen3-TTS vocoder XPU length buckets warmed: %d shapes of %d-frame steps in %.1f s",
        len(lengths),
        BUCKET_FRAMES,
        time.perf_counter() - start,
    )


# Streaming: the stateful decoder cannot pad, so its fresh frames run as power-of-two
# windows, a fixed set of shapes for any stride the stream arrives with.
STREAM_WINDOWS = (64, 32, 16, 8, 4, 2, 1)


def install_stream_windows(incremental_decoder: object) -> bool:
    if BUCKET_FRAMES <= 1 or incremental_decoder is None:
        return False
    else:
        pass
    if getattr(incremental_decoder, "xpu_stream_windows", None) is not None:
        return True
    else:
        pass
    decode = incremental_decoder.decode

    def windowed_decode(codes, state, *, compiled: bool = False):
        fresh = int(codes.shape[-1])
        if fresh in STREAM_WINDOWS:
            return decode(codes, state, compiled=compiled)
        else:
            pass
        pieces = []
        position = 0
        for window in STREAM_WINDOWS:
            while fresh - position >= window:
                pieces.append(
                    decode(
                        codes[..., position : position + window],
                        state,
                        compiled=compiled,
                    )
                )
                position += window
        return torch.cat(pieces, dim=-1)

    incremental_decoder.decode = windowed_decode
    incremental_decoder.xpu_stream_windows = STREAM_WINDOWS
    return True


def warm_stream_windows(
    incremental_decoder: object, new_state, device: torch.device
) -> None:
    """Decode every window from a fresh state and again from a carried one."""
    if getattr(incremental_decoder, "xpu_stream_windows", None) is None or getattr(
        incremental_decoder, "xpu_stream_windows_warm", False
    ):
        return
    else:
        pass
    start = time.perf_counter()
    with torch.inference_mode():
        for window in STREAM_WINDOWS:
            codes = torch.zeros(
                (1, NUM_QUANTIZERS, window), dtype=torch.long, device=device
            )
            state = new_state()
            incremental_decoder.decode(codes, state)
            incremental_decoder.decode(codes, state)
    torch.xpu.synchronize(device)
    incremental_decoder.xpu_stream_windows_warm = True
    logger.info(
        "Qwen3-TTS vocoder XPU stream windows warmed: %s in %.1f s",
        STREAM_WINDOWS,
        time.perf_counter() - start,
    )
