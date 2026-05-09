"""
Legacy API compatibility mixin for pyxdaq < 0.6.0.

Defines ``_LegacyMixin`` which is mixed into ``XDAQ`` directly, so the
deprecated camelCase methods are available on every ``XDAQ`` instance.
Each deprecated method emits a ``DeprecationWarning`` and delegates to
the current snake_case implementation.

Only methods that appeared in published v0.5.x examples are shimmed here.
Will be removed in a future major release.
"""

import warnings
from typing import TYPE_CHECKING

from .datablock import DataBlock

if TYPE_CHECKING:
    from .xdaq import XDAQ


def _deprecated(old: str, new: str):
    """Emit a DeprecationWarning pointing at the caller's frame."""
    warnings.warn(
        f"{old}() is deprecated and will be removed in a future release. "
        f"Use {new}() instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class _LegacyMixin:
    """Mixin that adds pre-0.6 camelCase aliases onto XDAQ."""

    # ── Sample rate ───────────────────────────────────────────────────────

    def getSampleSizeBytes(self: "XDAQ"):
        _deprecated("getSampleSizeBytes", "sample_size_in_bytes")
        return self.sample_size_in_bytes()

    # ── Data acquisition ──────────────────────────────────────────────────

    def runAndReadDataBlock(self: "XDAQ", samples) -> DataBlock:
        _deprecated("runAndReadDataBlock", "acquire_samples")
        return DataBlock.from_buffer(
            self.rhs,
            self.sample_size_in_bytes(),
            self.acquire_raw_data(samples),
            self.num_enabled_datastream,
            self.device_timestamp,
        )

    # ── Stim ──────────────────────────────────────────────────────────────

    def setStimCmdMode(self: "XDAQ", enabled: bool):
        _deprecated("setStimCmdMode", "stim.enable/stim.disable")
        if enabled:
            self.stim.enable()
        else:
            self.stim.disable()

    def manual_trigger(self: "XDAQ", trigger: int, enable: bool):
        _deprecated("manual_trigger", "stim.trigger")
        self.stim.trigger(trigger, enable)

    # ── Data streams ──────────────────────────────────────────────────────

    def enableDataStream(self: "XDAQ", stream, enable: bool, force=False):
        _deprecated("enableDataStream", "config_data_stream")
        return self.config_data_stream(stream, enable, force)

    @property
    def numDataStream(self: "XDAQ"):
        _deprecated("numDataStream", "num_enabled_datastream")
        return self.num_enabled_datastream

    # ── TTL ───────────────────────────────────────────────────────────────

    def setTTLout(self: "XDAQ", channel, enable):
        _deprecated("setTTLout", "set_ttl_out")
        return self.set_ttl_out(channel, enable)
