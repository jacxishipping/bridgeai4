"""Optional WASAPI loopback capture for Live Desktop audio (opt-in).

Default remains off (`AI_BRIDGE_DESKTOP_AUDIO` unset). When enabled on Windows,
we prefer real WASAPI speaker loopback via ``pyaudiowpatch`` (with automatic
channel/rate normalization to 48kHz stereo s16le), then fall back to the
optional ``soundcard`` package; if neither backend or loopback endpoint is
available, callers fall back honestly to silent PCM frames.
"""
from __future__ import annotations

import logging
import os
import queue
import struct
import threading
import time
from typing import Any

log = logging.getLogger("ai_bridge.desktop_audio")

_PROBE_CACHE: dict[str, Any] | None = None


def desktop_audio_opted_in() -> bool:
    return os.environ.get("AI_BRIDGE_DESKTOP_AUDIO", "").strip().lower() in ("1", "true", "yes")


def wasapi_capture_available() -> bool:
    """True when this host can open a WASAPI/speaker loopback device."""
    info = probe_wasapi_loopback()
    return bool(info.get("available"))


def get_desktop_audio_status() -> dict[str, Any]:
    """Structured status for operator UI & explicit audio consent banners."""
    opted_in = desktop_audio_opted_in()
    probe = probe_wasapi_loopback()
    return {
        "enabled": opted_in,
        "live_loopback_available": bool(probe.get("available")),
        "backend": probe.get("backend") if (opted_in and probe.get("available")) else ("silent_pcm" if opted_in else "disabled"),
        "device": probe.get("device"),
        "consent_required": True,
        "reason": probe.get("reason") if opted_in else "disabled_by_default",
    }


def _resample_s16_stereo(raw_bytes: bytes, src_rate: int, src_channels: int, dst_rate: int, dst_frames: int) -> bytes:
    """Normalize arbitrary s16le PCM to stereo ``dst_rate`` with ``dst_frames`` samples."""
    if not raw_bytes or src_channels <= 0 or src_rate <= 0:
        return b"\x00" * (dst_frames * 2 * 2)
    sample_count = len(raw_bytes) // 2
    if sample_count < src_channels:
        return b"\x00" * (dst_frames * 2 * 2)
    samples = struct.unpack(f"<{sample_count}h", raw_bytes[: sample_count * 2])
    src_frames = sample_count // src_channels
    if src_rate == dst_rate and src_channels == 2 and src_frames == dst_frames:
        return raw_bytes[: dst_frames * 4]
    out: list[int] = []
    ratio = src_frames / float(max(1, dst_frames))
    for i in range(dst_frames):
        idx = min(src_frames - 1, int(i * ratio))
        base = idx * src_channels
        left = samples[base]
        right = samples[base + 1] if src_channels >= 2 else left
        out.append(left)
        out.append(right)
    return struct.pack(f"<{len(out)}h", *out)


def probe_wasapi_loopback() -> dict[str, Any]:
    """Probe once per process for Windows WASAPI loopback capability."""
    global _PROBE_CACHE
    if _PROBE_CACHE is not None:
        return dict(_PROBE_CACHE)
    result: dict[str, Any] = {
        "available": False,
        "backend": None,
        "device": None,
        "reason": "not_probed",
    }
    try:
        import platform

        if platform.system() != "Windows":
            result["reason"] = "wasapi_windows_only"
            _PROBE_CACHE = result
            return dict(result)
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"probe_failed:{type(exc).__name__}"
        _PROBE_CACHE = result
        return dict(result)

    # 1. Prefer pyaudiowpatch (native Windows WASAPI loopback)
    try:
        import pyaudiowpatch as pyaudio  # type: ignore

        p = pyaudio.PyAudio()
        try:
            dev = p.get_default_wasapi_loopback()
            if dev:
                result = {
                    "available": True,
                    "backend": "pyaudiowpatch_wasapi_loopback",
                    "device": str(dev.get("name") or "Default WASAPI Loopback"),
                    "sample_rate": int(dev.get("defaultSampleRate") or 48000),
                    "channels": max(1, int(dev.get("maxInputChannels") or 2)),
                    "reason": "ok",
                }
                _PROBE_CACHE = result
                return dict(result)
        finally:
            p.terminate()
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        log.debug("pyaudiowpatch WASAPI probe failed: %s", exc)

    # 2. Fallback to soundcard WASAPI loopback
    try:
        import soundcard as sc  # type: ignore

        mics = sc.all_microphones(include_loopback=True)
        loopbacks = [m for m in mics if getattr(m, "isloopback", False)]
        if not loopbacks:
            try:
                spk = sc.default_speaker()
                loopbacks = [sc.get_microphone(id=spk.id, include_loopback=True)]
            except Exception:
                loopbacks = []
        if not loopbacks:
            result["reason"] = "no_loopback_device"
            _PROBE_CACHE = result
            return dict(result)
        result = {
            "available": True,
            "backend": "soundcard_wasapi_loopback",
            "device": getattr(loopbacks[0], "name", None) or str(loopbacks[0]),
            "reason": "ok",
        }
    except ImportError:
        result["reason"] = "soundcard_or_pyaudiowpatch_not_installed"
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"probe_failed:{type(exc).__name__}"
        log.debug("WASAPI probe failed: %s", exc)
    _PROBE_CACHE = result
    return dict(result)


def reset_probe_cache() -> None:
    global _PROBE_CACHE
    _PROBE_CACHE = None


class WasapiLoopbackCapture:
    """Background speaker loopback capture yielding 48kHz stereo s16le PCM frames."""

    def __init__(self, sample_rate: int = 48000, channels: int = 2, samples_per_frame: int = 960):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.samples_per_frame = int(samples_per_frame)
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._backend = "none"
        self._device_name = ""
        self._error: str | None = None
        self._started = False

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def live(self) -> bool:
        return self._started and self._error is None and self._backend in (
            "pyaudiowpatch_wasapi_loopback",
            "soundcard_wasapi_loopback",
        )

    def _enqueue(self, pcm: bytes) -> None:
        try:
            self._q.put(pcm, timeout=0.05)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(pcm)
            except queue.Full:
                pass

    def _start_pyaudiowpatch(self) -> bool:
        try:
            import pyaudiowpatch as pyaudio  # type: ignore
        except ImportError:
            return False
        try:
            p = pyaudio.PyAudio()
            dev = p.get_default_wasapi_loopback()
            if not dev:
                p.terminate()
                return False
            src_rate = max(8000, int(dev.get("defaultSampleRate") or self.sample_rate))
            src_channels = max(1, min(2, int(dev.get("maxInputChannels") or 2)))
            src_frames = max(64, int(round(src_rate * (self.samples_per_frame / float(self.sample_rate)))))
            dev_index = int(dev["index"])
            self._device_name = str(dev.get("name") or "Default WASAPI Loopback")
            self._backend = "pyaudiowpatch_wasapi_loopback"
            self._stop.clear()

            def _run_wp() -> None:
                stream = None
                try:
                    stream = p.open(
                        format=pyaudio.paInt16,
                        channels=src_channels,
                        rate=src_rate,
                        input=True,
                        input_device_index=dev_index,
                        frames_per_buffer=src_frames,
                    )
                    while not self._stop.is_set():
                        raw = stream.read(src_frames, exception_on_overflow=False)
                        pcm = _resample_s16_stereo(raw, src_rate, src_channels, self.sample_rate, self.samples_per_frame)
                        self._enqueue(pcm)
                except Exception as exc:  # noqa: BLE001
                    self._error = f"{type(exc).__name__}:{exc}"
                    log.warning("pyaudiowpatch WASAPI loopback stopped: %s", exc)
                finally:
                    if stream is not None:
                        try:
                            stream.stop_stream()
                            stream.close()
                        except Exception:
                            pass
                    try:
                        p.terminate()
                    except Exception:
                        pass

            self._thread = threading.Thread(target=_run_wp, name="ai-bridge-wasapi-wp", daemon=True)
            self._thread.start()
            self._started = True
            deadline = time.time() + 1.5
            while time.time() < deadline and self._error is None:
                if not self._q.empty():
                    break
                time.sleep(0.05)
            return self._error is None
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}:{exc}"
            self._backend = "none"
            return False

    def start(self) -> bool:
        if self._started:
            return self.live
        self._error = None
        if self._start_pyaudiowpatch():
            return True
        self._error = None
        try:
            import numpy as np  # soundcard depends on numpy
            import soundcard as sc  # type: ignore
        except ImportError as exc:
            self._error = f"missing_dependency:{exc}"
            return False
        try:
            mics = sc.all_microphones(include_loopback=True)
            loopbacks = [m for m in mics if getattr(m, "isloopback", False)]
            if not loopbacks:
                spk = sc.default_speaker()
                loopbacks = [sc.get_microphone(id=spk.id, include_loopback=True)]
            mic = loopbacks[0]
            self._device_name = getattr(mic, "name", None) or str(mic)
            self._backend = "soundcard_wasapi_loopback"
            self._stop.clear()

            def _run() -> None:
                try:
                    with mic.recorder(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        blocksize=self.samples_per_frame,
                    ) as rec:
                        while not self._stop.is_set():
                            data = rec.record(numframes=self.samples_per_frame)
                            clipped = np.clip(data, -1.0, 1.0)
                            pcm = (clipped * 32767.0).astype("<i2").tobytes()
                            self._enqueue(pcm)
                except Exception as exc:  # noqa: BLE001
                    self._error = f"{type(exc).__name__}:{exc}"
                    log.warning("WASAPI loopback capture stopped: %s", exc)

            self._thread = threading.Thread(target=_run, name="ai-bridge-wasapi", daemon=True)
            self._thread.start()
            self._started = True
            deadline = time.time() + 1.5
            while time.time() < deadline and self._error is None:
                if not self._q.empty():
                    break
                time.sleep(0.05)
            return self._error is None
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}:{exc}"
            self._backend = "none"
            return False

    def read_pcm(self, silent_fallback: bytes) -> bytes:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return silent_fallback

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._thread = None
        self._started = False
