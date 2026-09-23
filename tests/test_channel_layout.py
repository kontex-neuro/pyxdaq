"""
Channel layout of an RHD2216, which fills only 16 of the 32 amplifier words its
datastream occupies. None of this needs a device attached.
"""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyxdaq.constants import HeadstageChipID, HeadstageChipMISOID, ZcheckPolarity
from pyxdaq.stream import RHDStreamer, RHSStreamer
from pyxdaq.xdaq import HdmiPort, StreamConfig, XDAQ


class FakeSamples:

    def __init__(self, amp):
        self.amp = amp


def fake_xdaq(streams, rhs=False):
    """An object exposing just the stream introspection part of XDAQ."""
    obj = type(
        'FakeXDAQ', (), {
            'rhs': rhs,
            'enabled_streams': property(lambda self: [s for s in streams if s.enabled]),
            'channels_per_stream_on_wire': XDAQ.channels_per_stream_on_wire,
            'enabled_stream_channels': XDAQ.enabled_stream_channels,
            'max_amp_channels_per_stream': XDAQ.max_amp_channels_per_stream,
        }
    )
    return obj()


def stream(chip, sid=0, miso=HeadstageChipMISOID.NA):
    return StreamConfig(available=True, chip=chip, miso=miso, sid=sid, enabled=True)


@pytest.mark.parametrize(
    'chip, miso, channels, channel_range', [
        (HeadstageChipID.RHD2132, HeadstageChipMISOID.NA, 32, "[ 0,31]"),
        (HeadstageChipID.RHD2216, HeadstageChipMISOID.NA, 16, "[ 0,15]"),
        (HeadstageChipID.RHD2164, HeadstageChipMISOID.MISO_A, 32, "[ 0,31]"),
        (HeadstageChipID.RHD2164, HeadstageChipMISOID.MISO_B, 32, "[32,63]"),
        (HeadstageChipID.RHS2116, HeadstageChipMISOID.NA, 16, "[ 0,15]"),
    ]
)
def test_stream_config_channels(chip, miso, channels, channel_range):
    s = stream(chip, miso=miso)
    assert s.num_channels == channels
    assert s.channel_range == channel_range


def test_undetected_stream_has_no_channels():
    s = StreamConfig(sid=0)
    assert s.num_channels == 0
    assert s.channel_range == "NA"


@pytest.mark.parametrize(
    'chip, total, active, streams, wire, differential', [
        (HeadstageChipID.RHD2132, 32, 32, 1, 32, False),
        (HeadstageChipID.RHD2216, 16, 16, 1, 32, True),
        (HeadstageChipID.RHD2164, 64, 32, 2, 32, False),
        (HeadstageChipID.RHS2116, 16, 16, 1, 16, False),
    ]
)
def test_chip_capabilities(chip, total, active, streams, wire, differential):
    capabilities = chip.capabilities
    assert capabilities is not None
    assert capabilities.num_channels == chip.num_channels() == total
    assert capabilities.channels_per_stream == chip.num_channels_per_stream() == active
    assert capabilities.streams_per_chip == streams
    assert capabilities.channels_per_stream_on_wire == chip.channels_per_stream_on_wire() == wire
    assert capabilities.differential_inputs is differential
    with pytest.raises(FrozenInstanceError):
        capabilities.channels_per_stream = 0


def test_unknown_chip_has_no_capabilities():
    assert HeadstageChipID.NA.capabilities is None
    assert HeadstageChipID.NA.num_channels() == 0
    assert HeadstageChipID.NA.num_channels_per_stream() == 0
    assert HeadstageChipID.NA.channels_per_stream_on_wire() == 0


@pytest.mark.parametrize(
    'chip, stream_ids, active_count, misos', [
        (HeadstageChipID.RHD2132, [8, 9], 1, [HeadstageChipMISOID.NA]),
        (HeadstageChipID.RHD2216, [8, 9], 1, [HeadstageChipMISOID.NA]),
        (
            HeadstageChipID.RHD2164, [8, 9
                                     ], 2, [HeadstageChipMISOID.MISO_A, HeadstageChipMISOID.MISO_B]
        ),
        (HeadstageChipID.RHS2116, [4], 1, [HeadstageChipMISOID.NA]),
        (HeadstageChipID.NA, [8, 9], 0, []),
        (HeadstageChipID.NA, [4], 0, []),
    ]
)
def test_discovery_preserves_stream_slots(chip, stream_ids, active_count, misos):
    port = HdmiPort.fromChipInfos([(3, chip)], [stream_ids], 2)
    assert port.portNumber == 2
    assert [config.sid for config in port.streams] == stream_ids
    for index, config in enumerate(port.streams):
        assert not config.enabled
        if index < active_count:
            assert config.available
            assert config.chip == chip
            assert config.delay == 3
            assert config.miso == misos[index]
        else:
            assert config == StreamConfig(sid=stream_ids[index])
    assert HdmiPort.from_json(port.to_json()) == port


def test_rhd2216_occupies_a_full_wire_slot():
    assert HeadstageChipID.RHD2216.num_channels_per_stream() == 16
    assert HeadstageChipID.RHD2216.channels_per_stream_on_wire() == 32
    assert HeadstageChipID.RHD2132.channels_per_stream_on_wire() == 32
    assert HeadstageChipID.RHS2116.channels_per_stream_on_wire() == 16


def test_enabled_stream_channels_reports_real_channels():
    x = fake_xdaq(
        [
            stream(HeadstageChipID.RHD2132, sid=0),
            stream(HeadstageChipID.RHD2216, sid=1),
            StreamConfig(sid=2),  # not enabled
        ]
    )
    assert x.enabled_stream_channels() == [32, 16]
    # the RHD2132 still needs a full 32 channel sweep
    assert x.max_amp_channels_per_stream() == 32


def test_board_of_only_rhd2216_sweeps_16_channels():
    x = fake_xdaq([stream(HeadstageChipID.RHD2216, sid=i) for i in range(2)])
    assert x.enabled_stream_channels() == [16, 16]
    assert x.max_amp_channels_per_stream() == 16


@pytest.mark.parametrize('rhs, width', [(False, 32), (True, 16)])
def test_stream_enabled_without_a_chip_keeps_full_width(rhs, width):
    """Manually enabled streams must not be silently dropped from recordings."""
    x = fake_xdaq([StreamConfig(sid=0, enabled=True)], rhs=rhs)
    assert x.enabled_stream_channels() == [width]
    assert x.max_amp_channels_per_stream() == width


def test_rhd2216_with_undetected_stream_sweeps_full_width():
    x = fake_xdaq([stream(HeadstageChipID.RHD2216), StreamConfig(sid=1, enabled=True)])
    assert x.enabled_stream_channels() == [16, 32]
    assert x.max_amp_channels_per_stream() == 32


@pytest.mark.parametrize('rhs, width', [(False, 32), (True, 16)])
def test_no_enabled_streams_keeps_default_sweep_width(rhs, width):
    x = fake_xdaq([], rhs=rhs)
    assert x.max_amp_channels_per_stream() == width


@pytest.mark.parametrize('streamer_class, width', [(RHDStreamer, 32), (RHSStreamer, 16)])
def test_streamer_channel_count_bounds(streamer_class, width):
    for count in (-1, width + 1):
        with pytest.raises(ValueError, match=f'between 0 and {width}'):
            streamer_class([count])

    streamer = streamer_class([0, width])
    amp = np.zeros((2, 2, width), dtype=np.uint16)
    assert streamer.num_channels() == width
    assert streamer._flatten(amp).shape == (2, width)


def test_streamer_drops_rhd2216_dummy_channels():
    amp = np.arange(2 * 3 * 32, dtype=np.uint16).reshape(2, 3, 32)
    s = RHDStreamer([32, 16, 32])
    out = s.get_amp_data_for_stream('continuous', FakeSamples(amp))

    assert s.num_channels() == 80
    assert out.shape == (2, 80)
    assert np.array_equal(out[:, :32], amp[:, 0, :])
    assert np.array_equal(out[:, 32:48], amp[:, 1, :16])
    assert np.array_equal(out[:, 48:], amp[:, 2, :])


def test_streamer_without_layout_keeps_every_wire_channel():
    amp = np.zeros((2, 3, 32), dtype=np.uint16)
    s = RHDStreamer()
    assert s.get_amp_data_for_stream('continuous', FakeSamples(amp)).shape == (2, 96)
    assert s.num_channels(3) == 96
    with pytest.raises(ValueError):
        s.num_channels()


def test_rhs_streamer_splits_ac_and_dc():
    amp = np.arange(2 * 2 * 16 * 2, dtype=np.uint16).reshape(2, 2, 16, 2)
    s = RHSStreamer([16, 16])
    assert s.num_channels() == 32
    assert np.array_equal(
        s.get_amp_data_for_stream('AC', FakeSamples(amp)), amp[..., 1].reshape(2, -1)
    )
    assert np.array_equal(
        s.get_amp_data_for_stream('DC', FakeSamples(amp)), amp[..., 0].reshape(2, -1)
    )


def test_streamer_rejects_wrong_stream_count():
    amp = np.zeros((2, 3, 32), dtype=np.uint16)
    with pytest.raises(ValueError, match='configured for 2 datastreams'):
        RHDStreamer([32, 32]).get_amp_data_for_stream('continuous', FakeSamples(amp))


def test_impedance_mask_marks_absent_channels():
    """The mask measure_impedance applies to a mixed RHD2132 / RHD2216 board."""
    test_channels = np.arange(32)
    stream_channels = np.array([32, 16])
    present = np.less.outer(test_channels, stream_channels).T

    assert present.shape == (2, 32)
    assert present[0].all()
    assert present[1, :16].all()
    assert not present[1, 16:].any()


def test_zcheck_polarity_sets_register_5_bit_1():
    from pyxdaq import resources
    from pyxdaq.constants import SampleRate
    from pyxdaq.rhd_driver import RHDDriver

    reg = RHDDriver(SampleRate.SampleRate30000Hz, resources.rhd.reg_path, resources.rhd.isa_path)

    reg.set_zcheck_polarity(ZcheckPolarity.Negative)
    assert (int(reg.controller.registers[5]) >> 1) & 1 == 1
    reg.set_zcheck_polarity(ZcheckPolarity.Positive)
    assert (int(reg.controller.registers[5]) >> 1) & 1 == 0
