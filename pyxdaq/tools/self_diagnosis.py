import json
import platform
import shutil
import argparse
from pathlib import Path
try:
    from rich.console import Console
    from rich.table import Table
    import rich.json
except ImportError as e:
    print('Project is not installed. See README.md for installation.')
    exit(1)

results = {
    'OS': platform.system(),
    'python': platform.python_version(),
}


def run_diagnosis(console: Console):
    console.print('Checking package installation...', end='')
    try:
        import pyxdaq
        import pylibxdaq
        from pylibxdaq import pyxdaq_device
        results['dependencies'] = 'OK'
        console.print('[OK]', style='bold green')
    except ImportError as e:
        results['dependencies'] = str(e)
        console.print(f'Project is not installed. See README.md for installation. {e}')
        return

    console.print('Checking XDAQ dynamic libraries...', end='')

    def try_load_device_manager(path: Path):
        try:
            manager = pyxdaq_device.get_device_manager(str(path))
            yield ('loading', 'OK')
        except Exception as e:
            yield ('loading_error', str(e))
            return

        try:
            info = manager.info()
            yield ('info', info)
        except Exception as e:
            yield ('info_error', str(e))
            return

        try:
            parsed_info = json.loads(info)
            yield ('name', parsed_info['name'])
        except Exception as e:
            yield ('info_parse_error', str(e))

    try:
        from pylibxdaq.managers import manager_paths
        results['Device Managers'] = {
            str(manager_path): {
                k: r
                for k, r in try_load_device_manager(manager_path)
            }
            for manager_path in manager_paths
        }
        names = [i.get('name') for i in results['Device Managers'].values() if 'name' in i]
        missing = []
        if 'XDAQ OpalKelly USB' not in names:
            missing.append('Gen1 (OpalKelly USB)')
        if 'XDAQ Thunderbolt' not in names:
            missing.append('Gen2 (Thunderbolt)')
        if missing:
            raise RuntimeError(f"Missing XDAQ libraries: {', '.join(missing)}")
        console.print('[OK]', style='bold green')
    except Exception as e:
        results['Device Managers'] = str(e)
        console.print(
            '[bold red]broken installation or unable to load XDAQ dynamic libraries!'
            'Please follow the instructions at README.md to re-install or contact KonteX support.[/bold red]'
        )
        return

    console.print('Detecting XDAQ...', end='')
    try:
        from pyxdaq.board import Board
        device_list = Board.list_devices()
        if len(device_list) == 0:
            raise RuntimeError("No XDAQ devices found.")
        for device in device_list:
            results[str(device)] = check_xdaq(console, device)

    except Exception as e:
        results['xdaq'] = str(e)
        console.print(f'Unable to detect XDAQ. {e}', style='bold red')
        return


def check_xdaq(console: Console, device):
    device_results = {}
    try:
        from pyxdaq.board import Board, DeviceSetup
        assert isinstance(device, DeviceSetup)

        with Board(device) as board:
            device_results['xdaq_info'] = str(board.dev.get_info())
            device_results['xdaq_status'] = str(board.dev.get_status())
            json.loads(device_results['xdaq_info'])
            json.loads(device_results['xdaq_status'])
            console.print('[OK]', style='bold green')
            console.print('XDAQ Info', style='blue')
            console.print(rich.json.JSON(device_results['xdaq_info']))
            console.print('XDAQ Status', style='blue')
            console.print(rich.json.JSON(device_results['xdaq_status']))
    except Exception as e:
        device_results['xdaq'] = str(e)
        console.print(f'Unable to detect XDAQ. {e}', style='bold red')
        return device_results

    console.print('\nDetecting Recording X-Headstages ...', end='')

    try:
        from pyxdaq.xdaq import XDAQ
        from pyxdaq.constants import SampleRate
        with Board(device.with_mode('rhd')) as board:
            device_results['fpga_rhd'] = board.dev.get_status()
            xdaq = XDAQ(board)
            xdaq.initialize()
            xdaq.changeSampleRate(SampleRate.SampleRate30000Hz)
            xdaq.findConnectedAmplifiers()

            tb = Table(
                show_header=True,
                header_style="bold magenta",
                title="XDAQ HDMI Ports (Recording X-Headstages)"
            )
            for n, port in enumerate(xdaq.ports):
                channels = sum(s.chip.num_channels_per_stream() for s in port)
                tb.add_column(
                    f"Port {n+1} [{channels}] Channels", justify="left", style="cyan", width=30
                )
            for row in list(zip(*xdaq.ports)):
                tb.add_row(*map(str, row))
            device_results['xdaq_rhd'] = xdaq.ports.to_dict()

            console.print('[OK]', style='bold green')
            console.print(tb)
            console.print('Please check if the number of channels is correct.', style='bold yellow')
    except Exception as e:
        device_results['xdaq_rhd'] = str(e)
        console.print('Unable to read XDAQ HDMI Ports.', style='bold red')
        console.print(f'{e}', style='red')
        return device_results

    console.print('\nDetecting Stim-Record X-Headstages ...', end='')
    try:
        with Board(device.with_mode('rhs')) as board:
            device_results['fpga_rhs'] = board.dev.get_status()
            xdaq = XDAQ(board)
            xdaq.initialize()
            xdaq.changeSampleRate(SampleRate.SampleRate30000Hz)
            xdaq.findConnectedAmplifiers()

            tb = Table(
                show_header=True,
                header_style="bold magenta",
                title="XDAQ HDMI Ports (Stim-Record X-Headstages)"
            )
            for n, port in enumerate(xdaq.ports):
                channels = sum(s.chip.num_channels_per_stream() for s in port)
                tb.add_column(
                    f"Port {n+1} [{channels}] Channels", justify="left", style="cyan", width=30
                )
            for row in list(zip(*xdaq.ports)):
                tb.add_row(*map(str, row))
            device_results['xdaq_rhs'] = xdaq.ports.to_dict()
            console.print('[OK]', style='bold green')
            console.print(tb)
            console.print('Please check if the number of channels is correct.', style='bold yellow')
    except Exception as e:
        device_results['xdaq_rhs'] = str(e)
        console.print('Unable to read XDAQ HDMI ports.', style='bold red')
        return device_results
    return device_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-pause', action='store_true')
    args = parser.parse_args()

    console = Console(height=shutil.get_terminal_size().lines - 1, highlight=False)
    run_diagnosis(console)

    Path('diagnostic_report.json').write_text(json.dumps(results, indent=4))
    console.print('diagnostic_report.json is generated.')

    if not args.no_pause:
        input('Press any key to exit…')


if __name__ == '__main__':
    main()
