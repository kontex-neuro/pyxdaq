import os
import gc
from pathlib import Path
import time
import platform
import shutil
import json

diagnostic_report = {
    'OS':platform.system(),
    'python':platform.python_version(),
}

def main():
    print('Checking dependencies...', end='')
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console(height=shutil.get_terminal_size().lines - 1,highlight=False)
        import dataclass_wizard
        import numpy 
        diagnostic_report['dependencies'] = 'OK'
        console.print('[OK]', style='bold green')
    except ImportError as e:
        diagnostic_report['dependencies'] = str(e)
        print('project is not installed! Please follow the instructions at README.md to install it.')
        return

    console.print('Checking pyxdaq...', end='')
    try:
        import pyxdaq
        diagnostic_report['pyxdaq'] = 'OK'
        console.print('[OK]', style='bold green')
    except ImportError as e:
        diagnostic_report['pyxdaq'] = str(e)
        console.print(
            '[bold red]pyxdaq is not installed! Please follow the instructions at README.md to install it.[/bold red]'
        )
        return

    console.print('Check project configuration...', end='')
    file_to_check = [
        'config/isa_rhd.json',
        'config/isa_rhs.json',
        'config/reg_rhd.json',
        'config/reg_rhs.json',
        'bitfiles/xr7310a75.bit',
        'bitfiles/xsr7310a75.bit',
    ]
    for f in file_to_check:
        if not Path(f).exists():
            diagnostic_report['project'] = f'{f} is missing!'
            console.print(f'{f} is missing! Please make sure that you are running this script from the root directory of the project.', style='bold red')
            return
    diagnostic_report['project'] = 'OK'
    console.print('[OK]', style='bold green')

    from pyxdaq import ok
    console.print('Checking Opal Kelly FrontPanel API...', end='')
    if hasattr(ok, 'is_mock'):
        if platform.system() == 'Windows':
            driver_url = 'https://pins.opalkelly.com/downloads'
            console.print(
                'Opal Kelly FrontPanel API is not installed or not working properly.'
                f'\nPlease make sure that the driver is installed from [link={driver_url}]Opal Kelly website[/link].'
            ,style='bold red')
        else:
            console.print(
                'Opal Kelly FrontPanel API is not working properly.', style='bold red'
            )
        diagnostic_report['frontpanel'] = 'Not working properly' 
        return
    else:
        console.print('[OK]', style='bold green')
        diagnostic_report['frontpanel'] = 'OK'

    console.print('\nDetecting XDAQ...')
    try:
        from pyxdaq.board import OkBoard
        from pyxdaq.xdaq import XDAQModel,XDAQ

        def get_device():
            dev = ok.okCFrontPanel()
            devices = [(dev.GetDeviceListModel(i), dev.GetDeviceListSerial(i)) for i in range(dev.GetDeviceCount())]
            diagnostic_report['ok_devices'] = devices
            for md, sn in devices:
                console.print(f'Found Opal Kelly Device: [bold]{md}[/bold] [bold]{sn}[/bold]')
                if md == ok.okPRODUCT_XEM7310A75:
                    console.print(f'Trying to open {sn} as XDAQ...', end='')
                    res = dev.OpenBySerial(sn)
                    if res != ok.okCFrontPanel.NoError:
                        console.print(f'[bold red]Open failed {res}[/bold red]')
                        continue
                    board = OkBoard(dev=dev)
                    try:
                        xdaq = XDAQ('config', dev=board)
                        xdaq.config_fpga(False, 'bitfiles/xr7310a75.bit')
                        return xdaq
                    except RuntimeError as e:
                        if e.args[0] == 'XDAQ MCU did not start in time':
                            console.print('[bold red]XDAQ MCU was not detected[/bold red]')
                            continue
                        else:
                            console.print(f'Unkown error: {e}', style='bold red')
                            continue
                    except Exception as e:
                        console.print(f'Unkown error: {e}', style='bold red')
                        continue

            raise RuntimeError('No supported device found')


        xdaq = get_device()
    
        diagnostic_report['xdaq_info'] = str(xdaq.xdaqinfo)
        diagnostic_report['xdaq'] = 'OK'
        console.print('[OK]', style='bold green')
        console.print(f'XDAQ Serial Number: [bold]{xdaq.xdaqinfo.serial:X}[/bold]', style='blue')
        console.print(f'XDAQ Model: [bold]{xdaq.xdaqinfo.model.name}[/bold]', style='blue')
        console.print(f'XDAQ Recording Channels: [bold]{xdaq.xdaqinfo.rhd}[/bold]', style='blue')
        console.print(f'XDAQ Stim-Record Channels: [bold]{xdaq.xdaqinfo.rhs}[/bold]', style='blue')
        console.print(f'XDAQ Expander: [bold]{xdaq.expander[0]}[/bold]', style='blue')
    except Exception as e:
        diagnostic_report['xdaq'] = str(e)
        console.print(f'Unable to detect XDAQ. {e}', style='bold red')
        return

    console.print('\nDetecting Recording X-Headstages ...', end='')

    try:
        from pyxdaq.constants import SampleRate
        xdaq.initialize()
        xdaq.changeSampleRate(SampleRate.SampleRate30000Hz, False)
        xdaq.findConnectedAmplifiers()

        tb = Table(show_header=True, header_style="bold magenta", title="XDAQ HDMI Ports (Recording X-Headstages)")
        rhd_ports = xdaq.xdaqinfo.rhd // 256 if xdaq.xdaqinfo.model == XDAQModel.One else xdaq.xdaqinfo.rhd // 128
        for n, port in enumerate(xdaq.ports):
            channels = sum(s.chip.num_channels_per_stream() for s in port)
            tb.add_column(f"Port {n+1} [{channels}] Channels", justify="left", style="cyan", width=30)
        for row in list(zip(*xdaq.ports))[:8 if xdaq.xdaqinfo.model == XDAQModel.One else 4]:
            tb.add_row(*map(str, row[:rhd_ports]))
        diagnostic_report['xdaq_rhd'] = str(xdaq.ports)
        console.print('[Done]', style='bold green')
        console.print(tb)
        console.print('Please check if the number of channels is correct.', style='bold yellow')
    except Exception as e:
        diagnostic_report['xdaq_rhd'] = str(e)
        console.print('Unable to read XDAQ HDMI Ports.', style='bold red')
        return

    console.print('\nDetecting Stim-Record X-Headstages ...', end='') 
    try:
        xdaq.config_fpga(True, 'bitfiles/xsr7310a75.bit')
        xdaq.initialize()
        xdaq.changeSampleRate(SampleRate.SampleRate30000Hz)
        xdaq.findConnectedAmplifiers()

        tb = Table(show_header=True, header_style="bold magenta", title="XDAQ HDMI Ports (Stim-Record X-Headstages)")
        rhs_ports = xdaq.xdaqinfo.rhs // 32
        for n, port in enumerate(xdaq.ports):
            channels = sum(s.chip.num_channels_per_stream() for s in port)
            tb.add_column(f"Port {n+1} [{channels}] Channels", justify="left", style="cyan", width=30)
        for row in list(zip(*xdaq.ports))[:2 if xdaq.xdaqinfo.rhs == 32 else 1]:
            tb.add_row(*map(str, row[:rhs_ports]))
        diagnostic_report['xdaq_rhs'] = str(xdaq.ports)
        console.print('[Done]', style='bold green')
        console.print(tb)
        console.print('Please check if the number of channels is correct.', style='bold yellow')
    except Exception as e:
        diagnostic_report['xdaq_rhs'] = str(e)
        console.print('Unable to read XDAQ HDMI Ports.', style='bold red')
        return


if __name__ == '__main__':
    main()
    with open('diagnostic_report.json','w') as f:
        json.dump(diagnostic_report,f,indent=4)
    print('diagnostic_report.json is generated.')