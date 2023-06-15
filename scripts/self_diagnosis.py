import os
import gc
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

    console.print('\nChecking XDAQ...', end='')
    try:
        from pyxdaq.xdaq import get_XDAQ
        from pyxdaq.xdaq import XDAQModel

        xdaq = get_XDAQ(bitfile='bitfiles/xr7310a75.bit')
    
        diagnostic_report['xdaq_info'] = str(xdaq.xdaqinfo)
        diagnostic_report['xdaq'] = 'OK'
        console.print('[OK]', style='bold green')
        console.print(f'XDAQ Serial Number: [bold]{xdaq.xdaqinfo.serial:X}[/bold]', style='blue')
        console.print(f'XDAQ Model: [bold]{xdaq.xdaqinfo.model.name}[/bold]', style='blue')
        console.print(f'XDAQ Recording Channels: [bold]{xdaq.xdaqinfo.rhd}[/bold]', style='blue')
        console.print(f'XDAQ Stim-Record Channels: [bold]{xdaq.xdaqinfo.rhs}[/bold]', style='blue')
    except Exception as e:
        diagnostic_report['xdaq'] = str(e)
        console.print('XDAQ is not working properly.', style='bold red')
        return

    console.print('\nDetecting Recording X-Headstages ...', end='')

    try:
        tb = Table(show_header=True, header_style="bold magenta", title="XDAQ HDMI Ports (Recording X-Headstages)")
        rhd_ports = xdaq.xdaqinfo.rhd // 256 if xdaq.xdaqinfo.model == XDAQModel.ONE else xdaq.xdaqinfo.rhd // 128
        for n, port in enumerate(xdaq.ports):
            channels = sum(s.chip.num_channels_per_stream() for s in port)
            tb.add_column(f"Port {n+1} [{channels}] Channels", justify="left", style="cyan", width=30)
        for row in list(zip(*xdaq.ports))[:8 if xdaq.xdaqinfo.model == XDAQModel.ONE else 4]:
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
        del xdaq
        gc.collect()
        time.sleep(0.5)
        xdaq = get_XDAQ(bitfile='bitfiles/xsr7310a75.bit', rhs=True)

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