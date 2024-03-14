#%%
from dataclasses import dataclass
from functools import partial
from pyxdaq.impedance import calculate_impedance, Frequency
from pyxdaq.datablock import amplifier2mv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple
from tqdm.auto import tqdm

raw_data = np.load('res.npz')
test_frequencies = raw_data['test_frequencies']
test_channels = raw_data['test_channels']
raw_measurements = [raw_data[f'arr_{i}'] for i in range(len(test_frequencies))]
sample_rate = raw_data['sample_rate'][0]
capacitor = np.array([0.1e-12, 1e-12, 10e-12])
periods = raw_data['periods']
# the last dimension is the signal length, might be different for each frequency
num_runs, num_caps, num_streams, num_chs, _ = raw_measurements[0].shape


def get_impedance(raw_measurements, sample_rate, test_frequencies, periods):

    def get_impedance_for_runs(raw_at_f: np.ndarray, frequency: float, period: int):
        # Runs, Caps, Streams, Channels, Signal
        runs, caps, streams, chs, siglen = raw_at_f.shape
        # Runs, Caps, Streams * Channels, Signal
        raw_at_f = raw_at_f.reshape((runs, caps, -1, siglen))
        # Caps, Runs, Streams * Channels, Signal
        raw_at_f = raw_at_f.transpose((1, 0, 2, 3))
        # Caps, Runs * Streams * Channels, Signal
        raw_at_f = raw_at_f.reshape((caps, -1, siglen))
        # [magnitude, phase, capacitor] x [Runs * Streams * Channels]
        return np.array(
            calculate_impedance(
                all_signals=raw_at_f[:, :, 3 + period * 2:3 - period],
                sample_rate=sample_rate,
                rhs=True,
                frequency=frequency,
                return_cap=True,
            )
        ).reshape((3, runs, streams, chs))

    return np.array(
        [
            get_impedance_for_runs(raw_at_f, frequency, period) for raw_at_f, frequency,
            period in tqdm(zip(raw_measurements, test_frequencies, periods))
        ]
    )


# frequency, [magnitude, phase, capacitor], stream, channel
impedance_results = get_impedance(raw_measurements, sample_rate, test_frequencies, periods)
# frequency, [magnitude, phase, capacitor], stream * channel
impedance_results = impedance_results.reshape((*impedance_results.shape[:-2], -1))


#%%
def format_unit(value: float) -> Tuple[float, str]:
    if value >= 1e6:
        value, unit = value / 1e6, 'M'
    elif value >= 1e3:
        value, unit = value / 1e3, 'k'
    elif value >= 1:
        unit = ''
    elif value >= 1e-3:
        value, unit = value * 1e3, 'm'
    elif value >= 1e-6:
        value, unit = value * 1e6, 'µ'
    elif value >= 1e-9:
        value, unit = value * 1e9, 'n'
    else:
        value, unit = value * 1e12, 'p'

    return value, unit


def print_impedance(magnitude, phase, expected_magnitude=None, expected_phase=None):
    if expected_magnitude is None:
        for ch, (m, p) in enumerate(zip(magnitude, phase)):
            m, unit = format_unit(m)
            print(f'{ch:2d} {m:7.3f} {unit}Ω {p:7.3f}°')
    else:
        for ch, (m, p, em, ep) in enumerate(zip(magnitude, phase, expected_magnitude,
                                                expected_phase)):
            m, unit = format_unit(m)
            em, eunit = format_unit(em)
            print(f'{ch:2d} {m:7.3f} {unit}Ω {p:7.3f}° | {em:7.3f} {eunit}Ω {ep:7.3f}°')


@dataclass
class ImpedanceTestModule:
    """
    The resistance and capacitance of the impedance test module
    """
    test_channels: np.ndarray

    board_R = np.ones(16) * 10e6
    board_C = np.array([[27e-12, 47e-12, 75e-12, 100e-12, 150e-12, 300e-12, 1.5e-9, 15e-9]] * 2
                      ).flatten()
    channel_R: np.ndarray = None
    channel_C: np.ndarray = None

    is_rhs = True
    n_headstages = 1
    _streams_per_x3sr32_headstage = 2

    @staticmethod
    def RC_parallel_magnitude(f, R, C):
        return 1 / np.sqrt((1 / R)**2 + (2 * np.pi * f * C)**2)

    @staticmethod
    def RC_parallel_phase(f, R, C):
        return -np.arctan(2 * np.pi * f * R * C)  # in radians

    def __post_init__(self):
        if not self.is_rhs or self.n_headstages != 1:
            raise NotImplementedError('Only 1 X3SR32 headstage is supported for now')
        self.channel_R = np.concatenate(
            [self.board_R[self.test_channels]] * self._streams_per_x3sr32_headstage
        )
        self.channel_C = np.concatenate(
            [self.board_C[self.test_channels]] * self._streams_per_x3sr32_headstage
        )

    def calculate_expected_impedance(self, frequency) -> Tuple[np.ndarray, np.ndarray]:
        expected_magnitude = self.RC_parallel_magnitude(
            frequency.reshape((-1, 1)), self.channel_R.reshape((1, -1)),
            self.channel_C.reshape((1, -1))
        )
        expected_phase = np.rad2deg(
            self.RC_parallel_phase(
                frequency.reshape((-1, 1)), self.channel_R.reshape((1, -1)),
                self.channel_C.reshape((1, -1))
            )
        )
        expected_magnitude = np.concatenate(
            [expected_magnitude] * self._streams_per_x3sr32_headstage, axis=1
        )
        expected_phase = np.concatenate(
            [expected_phase] * self._streams_per_x3sr32_headstage, axis=1
        )
        return expected_magnitude, expected_phase


impedance_module = ImpedanceTestModule(test_channels)
# compute the expected impedance under tested frequencies
# [Frequency, Channel], [Frequency, Channel]
expected_magnitude, expected_phase = impedance_module.calculate_expected_impedance(test_frequencies)

# compute the expected impedance under various frequencies for plotting
plot_target_freqs = np.logspace(np.log10(1), np.log10(10000), 100)
plot_target_expected_magnitude, plot_target_expected_phase = impedance_module.calculate_expected_impedance(
    plot_target_freqs
)
plot_actual_freqs = Frequency(plot_target_freqs).get_actual(sample_rate, display_warning=False)
plot_actual_expected_magnitude, plot_actual_expected_phase = impedance_module.calculate_expected_impedance(
    plot_actual_freqs
)


def plot_impedance_target_vs_actual():
    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    ax = axs[0]
    ax.set_title('Target Frequency vs Actual Frequency')
    ax.set_xlabel('Target Frequency')
    ax.set_ylabel('Actual Frequency')
    ax.plot(plot_target_freqs, plot_actual_freqs, label='Actual Frequency', color='C0')
    ax = axs[1]
    ax.set_title('Ideal Impedance under Target vs Actual Frequency')
    ax.set_xlabel('Target Frequency')
    ax.set_ylabel('Magnitude')
    ch = 0
    l1 = ax.plot(
        plot_target_freqs,
        plot_target_expected_magnitude[:, ch],
        alpha=0.7,
        label='Magnitude @ Target Frequency',
        color='C0'
    )

    l2 = ax.plot(
        plot_target_freqs,
        plot_actual_expected_magnitude[:, ch],
        alpha=0.7,
        label='Magnitude @ Actual Frequency',
        color='C1'
    )
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax2 = ax.twinx()
    ax2.set_ylabel('Phase')
    l3 = ax2.plot(
        plot_target_freqs,
        plot_target_expected_phase[:, ch],
        alpha=0.7,
        label='Phase @ Target Frequency',
        color='C2'
    )
    l4 = ax2.plot(
        plot_target_freqs,
        plot_actual_expected_phase[:, ch],
        alpha=0.7,
        label='Phase @ Actual Frequency',
        color='C3'
    )
    ax2.set_yscale('linear')
    ax.legend(handles=[*l1, *l2, *l3, *l4], loc='lower left')
    fig.tight_layout()
    return fig


fig = plot_impedance_target_vs_actual()
plt.show()


#%%
def plot_impedance_test_results_with_impedance_test_module_attached():
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 14))
    fig.suptitle('Impedance Test with Test Module')
    # combine impedance results because there are only 8 different RC combinations
    #  for the test module
    combined_impedance_results = impedance_results.reshape((*impedance_results.shape[:-2], -1, 8))

    for ch, ax in enumerate(axs.T.flatten()):
        cap, capu = format_unit(impedance_module.channel_C[ch])
        r, ru = format_unit(impedance_module.channel_R[ch])
        ax.set_title(f'Channel {ch} {cap:.1f} {capu}F {r:.1f} {ru}Ω')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Magnitude')
        ax.set_xscale('log')
        ax.set_yscale('log')
        mag2plot = combined_impedance_results[:, 0, :, ch]
        for i in range(mag2plot.shape[1]):
            ax.scatter(test_frequencies, mag2plot[:, i], s=3, color='green')
        for i in range(8):
            if i == ch:
                l1 = ax.plot(
                    plot_target_freqs,
                    plot_actual_expected_magnitude[:, ch],
                    label='Expected Magnitude',
                    color='darkgreen',
                    alpha=0.5
                )
            else:
                ax.plot(
                    plot_target_freqs,
                    plot_target_expected_magnitude[:, i],
                    color='lightgreen',
                    alpha=0.2
                )

        ax2 = ax.twinx()
        ax2.set_ylabel('Phase')
        ax2.set_yscale('linear')
        phase2plot = combined_impedance_results[:, 1, :, ch]
        for i in range(phase2plot.shape[1]):
            ax2.scatter(test_frequencies, phase2plot[:, i], s=3, color='blue')
        l2 = ax2.plot(
            plot_target_freqs,
            plot_actual_expected_phase[:, ch],
            label='Expected Phase',
            color='darkblue',
            alpha=0.5
        )
        ax.legend(handles=[l1[0], l2[0]], loc='lower left')
        ax.set_xlim(
            np.min(test_frequencies) - 10,
            np.max(test_frequencies) + 1000,
        )

    fig.tight_layout()
    return fig


fig = plot_impedance_test_results_with_impedance_test_module_attached()
plt.show()


#%%
def plot_test_signal_and_measured_waveform():
    freq_idx = 1
    ch_idx = 0
    # take run 0
    period = periods[freq_idx]
    frequency = test_frequencies[freq_idx]
    raw = raw_measurements[freq_idx].reshape((num_runs, num_caps, num_streams * num_chs, -1))[0]

    m, p, c = calculate_impedance(
        # fpga induced a 3 sample delay
        all_signals=raw[:, :, 3 + period * 2:3 - period],
        sample_rate=sample_rate,
        rhs=True,
        frequency=test_frequencies[freq_idx],
        return_cap=True,
    )
    start, end = 2 * period, -period
    sig = raw[c[ch_idx], ch_idx, 3 + start:3 + end]
    sig = amplifier2mv(sig) / 1000

    fig, ax = plt.subplots(1, 1, figsize=(4, 2))
    cap_v, cap_u = format_unit(capacitor[c[ch_idx]])
    ax.set_title(f'@ {test_frequencies[freq_idx]:.0f} Hz {cap_v} {cap_u}F')
    ax.plot(sig)
    ax = ax.twinx()

    wave = np.sin(2 * np.pi * np.arange(0, raw.shape[-1]) / period)
    ax.plot(wave[start:end], color='red', alpha=0.5)
    return fig


fig = plot_test_signal_and_measured_waveform()
plt.show()


#%%
def plot_3d_surface():
    # plot 3d surface
    # x: RC values
    # y: frequency
    # z1: magnitude, z2: expected magnitude
    X = np.arange(8)
    Y = test_frequencies
    XX, YY = np.meshgrid(X, Y)
    # Frequency, [magnitude, phase, capacitor], run, channel
    # Frequency, [magnitude, phase, capacitor], run * 4, 8
    # Frequency, [magnitude, phase, capacitor], 8
    mean_merged_impedance_results = impedance_results.reshape(
        (*impedance_results.shape[:-2], -1, 8)
    ).mean(axis=-2)

    def format_rc(R, C):
        R, unit = format_unit(R)
        C, cunit = format_unit(C)
        return f'{C:.0f}{cunit}F'

    X_label = [
        format_rc(r, c) for r, c in zip(
            impedance_module.board_R[:8],
            impedance_module.board_C[:8],
        )
    ]
    c1, c2 = 'summer', 'cool'
    c1, c2 = 'Blues', 'Greens'

    def plot_3d_surface(ax, ax_contour, Z1, Z2, name):
        ax.set_xlabel('RC')
        ax.set_ylabel('Frequency')
        ax.set_zlabel(name)
        ax.set_xticks(X)
        ax.set_xticklabels(X_label)
        ax.set_zscale('log')
        surf = ax.plot_surface(XX, YY, Z1, cmap=c1, label=f'Measured {name}')
        surf = ax.plot_surface(XX, YY, Z2, cmap=c2, label=f'Expected {name}')
        error = np.abs(Z1 - Z2)
        surf = ax.plot_surface(XX, YY, error, cmap='Reds', label='Error')
        ax.legend(loc='upper left')

        if ax_contour is None:
            return

        # plot contour for the error
        ax_contour.set_xticks(X)
        ax_contour.set_xticklabels(X_label)
        ax_contour.set_yscale('log')
        CS = ax_contour.contourf(XX, YY, error, cmap='Reds')
        ax_contour.set_title('Error')
        ax_contour.set_xlabel('RC')
        ax_contour.set_ylabel('Frequency')
        CS1 = ax_contour.contour(CS, levels=CS.levels, colors='b')
        return CS, CS1
        # add color bar

    return partial(
        plot_3d_surface,
        Z1=mean_merged_impedance_results[:, 0, :],
        Z2=expected_magnitude[:, :8],
        name='Magnitude',
    ), partial(
        plot_3d_surface,
        Z1=mean_merged_impedance_results[:, 1, :],
        Z2=expected_phase[:, :8],
        name='Phase',
    )


plot_mag_surface, plot_phase_surface = plot_3d_surface()

fig, [ax3d, axerr] = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.subplots_adjust(left=0, right=1)
ax3d.remove()
ax3d = fig.add_subplot(121, projection='3d')
cs, cs1 = plot_mag_surface(ax3d, axerr)
cbar = fig.colorbar(cs, ax=axerr, orientation='vertical')
plt.tight_layout()
plt.show()

fig, [ax3d, axerr] = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.subplots_adjust(left=0, right=1)
ax3d.remove()
ax3d = fig.add_subplot(121, projection='3d')
cs, cs1 = plot_phase_surface(ax3d, axerr)
cbar = fig.colorbar(cs, ax=axerr, orientation='vertical')
plt.tight_layout()
plt.show()
