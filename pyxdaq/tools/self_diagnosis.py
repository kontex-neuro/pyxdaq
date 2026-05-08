import json
import platform
import argparse
from pathlib import Path
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich import box
except ImportError as e:
    print('Missing dependency: rich. Install with: pip install pyxdaq[diagnosis]')
    exit(1)

results = {
    'OS': platform.system(),
    'python': platform.python_version(),
}

PASS = Text("✓ PASS", style="bold green")
FAIL = Text("✗ FAIL", style="bold red")
WARN = Text("⚠ WARN", style="bold yellow")


def step(console: Console, msg: str):
    """Print a step header."""
    console.print(f"\n[bold cyan]{'─' * 3} {msg}[/bold cyan]")


def run_diagnosis(console: Console):
    console.print(
        Panel(
            "[bold]XDAQ Self-Diagnosis[/bold]\n"
            f"OS: {platform.system()} {platform.machine()}  │  "
            f"Python: {platform.python_version()}",
            box=box.ROUNDED,
            style="blue",
        )
    )

    # ─── Step 1: Package installation ───
    step(console, "Package Installation")
    try:
        import pyxdaq
        import pylibxdaq
        from pylibxdaq import pyxdaq_device
        from importlib.metadata import version
        pkg_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        pkg_table.add_column("Package", style="white")
        pkg_table.add_column("Version", style="dim")
        pkg_table.add_column("Status")
        for pkg in ('pyxdaq', 'pylibxdaq'):
            pkg_table.add_row(pkg, version(pkg), PASS)
        console.print(pkg_table)
        results['dependencies'] = 'OK'
    except ImportError as e:
        results['dependencies'] = str(e)
        console.print(f"  {FAIL}  {e}")
        console.print("  [dim]Install with: pip install pyxdaq[/dim]")
        return

    # ─── Step 2: Device manager libraries ───
    step(console, "Device Manager Libraries")

    def try_load_device_manager(path: Path):
        try:
            manager = pyxdaq_device.get_device_manager(str(path))
        except Exception as e:
            return {'loading_error': str(e)}
        result = {'loading': 'OK'}
        try:
            info = manager.info()
            parsed = json.loads(info)
            result['name'] = parsed.get('name', '?')
            result['description'] = parsed.get('description', '')
        except Exception as e:
            result['info_error'] = str(e)
        return result

    try:
        from pylibxdaq.managers import DeviceManagerDir
        manager_paths = [
            p for p in DeviceManagerDir.iterdir()
            if p.suffix in ('.dylib', '.so', '.dll') and p.stem.endswith('_device_manager')
        ]
        mgr_results = {}
        mgr_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        mgr_table.add_column("Driver", style="white")
        mgr_table.add_column("Status")
        for mp in manager_paths:
            r = try_load_device_manager(mp)
            mgr_results[str(mp)] = r
            name = r.get('name', mp.stem)
            if 'loading_error' in r:
                mgr_table.add_row(name, FAIL)
            else:
                mgr_table.add_row(name, PASS)
        results['Device Managers'] = mgr_results
        console.print(mgr_table)

        names = [r.get('name') for r in mgr_results.values() if 'name' in r]
        missing = []
        if 'XDAQ OpalKelly USB' not in names:
            missing.append('Gen1 (OpalKelly USB)')
        if 'XDAQ Thunderbolt' not in names:
            missing.append('Gen2 (Thunderbolt)')
        if missing:
            for m in missing:
                console.print(f"  [yellow]⚠ {m} driver not found[/yellow]")
    except Exception as e:
        results['Device Managers'] = str(e)
        console.print(f"  {FAIL}  Unable to load device managers: {e}")
        console.print("  [dim]Reinstall pylibxdaq or contact KonteX support.[/dim]")
        return

    # ─── Step 3: Device enumeration ───
    step(console, "Connected Devices")
    try:
        from pyxdaq.board import Board
        device_list = Board.list_devices()
        if len(device_list) == 0:
            console.print(f"  {FAIL}  No XDAQ devices found.")
            console.print("  [dim]Check USB/Thunderbolt connections and power.[/dim]")
            results['devices'] = []
            return
        console.print(f"  Found [bold]{len(device_list)}[/bold] device(s)\n")
        results['devices'] = []
        for i, device in enumerate(device_list):
            device_result = check_xdaq(console, device, index=i + 1)
            results['devices'].append(device_result)
    except Exception as e:
        results['devices'] = str(e)
        console.print(f"  {FAIL}  {e}")
        return

    # ─── Summary ───
    console.print()
    total_devices = len(results.get('devices', []))
    ok_devices = sum(
        1 for d in results.get('devices', []) if isinstance(d, dict) and 'error' not in d
    )
    if ok_devices == total_devices:
        console.print(
            Panel(
                f"[bold green]All {total_devices} device(s) operational.[/bold green]",
                box=box.ROUNDED,
                style="green"
            )
        )
    else:
        console.print(
            Panel(
                f"[bold yellow]{ok_devices}/{total_devices} device(s) operational.[/bold yellow]",
                box=box.ROUNDED,
                style="yellow"
            )
        )


def check_xdaq(console: Console, device, index: int):
    from pyxdaq.board import BoardInfo
    assert isinstance(device, BoardInfo)

    device_results = {}

    # Open device and get basic info
    try:
        with device.create() as board:
            raw_info = board.raw.get_info()
            raw_status = board.raw.get_status()
            info = json.loads(raw_info or "{}")
            status = json.loads(raw_status or "{}")
            device_results['info'] = info
            device_results['status'] = status
    except Exception as e:
        device_results['error'] = str(e)
        console.print(
            Panel(
                f"[red]Unable to open device: {e}[/red]",
                title=f"[bold]Device {index}[/bold]",
                border_style="red",
            )
        )
        return device_results

    # Build device info panel
    model = info.get('XDAQ Model', '?')
    serial = info.get('Serial Number', '?')
    fpga = info.get('FPGA Vender', '?')
    fw_version = status.get('Version', '?')
    fw_build = status.get('Build', '?')
    fw_date = status.get('Date', '?')
    api_version = status.get('API', '?')
    mode = status.get('Mode', '?')
    expander = "Yes" if status.get('Expander') else "No"
    rhd_ch = info.get('RHD', '?')
    rhs_ch = info.get('RHS', '?')
    flash = info.get('Flash Memory', '')
    device_id = info.get('ID', '')
    np = info.get('NP', '')

    interface = "Thunderbolt" if 'thor' in str(device.manager_path) else "USB (OpalKelly)"
    gen = "Gen2" if "Thunderbolt" in interface else "Gen1"

    info_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=True)
    info_table.add_column("Key", style="dim", width=16)
    info_table.add_column("Value", style="white")
    info_table.add_row("Interface", f"{interface} ({gen})")
    info_table.add_row("Serial", serial)
    if device_id:
        info_table.add_row("Device ID", device_id)
    info_table.add_row("FPGA", fpga)
    if flash:
        info_table.add_row("Flash Memory", flash)
    info_table.add_row("Firmware", fw_version)
    info_table.add_row("Build", fw_build)
    info_table.add_row("FPGA Date", fw_date)
    info_table.add_row("API", api_version)
    info_table.add_row("Current Mode", mode.upper())
    info_table.add_row("Expander", expander)
    info_table.add_row("Max Channels", f"RHD: {rhd_ch}  │  RHS: {rhs_ch}")
    if np:
        info_table.add_row("NP Ports", str(np))

    capabilities = status.get('Capabilities', {})
    if capabilities:
        caps = ", ".join(
            f"{k}: {'/'.join(v) if isinstance(v, list) else v}" for k, v in capabilities.items()
        )
        info_table.add_row("Capabilities", caps)

    console.print(
        Panel(
            info_table,
            title=f"[bold white]Device {index}: XDAQ {model}[/bold white]",
            subtitle=f"[dim]{serial}[/dim]",
            border_style="bright_blue",
            box=box.ROUNDED,
        )
    )

    # ─── Headstage detection ───
    device_results['headstages'] = {}
    _detect_headstages(console, device, device_results, 'rhd', "Recording (RHD)")
    _detect_headstages(console, device, device_results, 'rhs', "Stim-Record (RHS)")

    return device_results


def _detect_headstages(console: Console, device, device_results: dict, mode: str, label: str):
    from pyxdaq.xdaq import XDAQ
    from pyxdaq.constants import SampleRate

    try:
        with device.with_mode(mode).create() as board:
            xdaq = XDAQ(board)
            xdaq.initialize()
            xdaq.changeSampleRate(SampleRate.SampleRate30000Hz)
            xdaq.find_connected_headstages()

            # Collect connected headstages per port
            port_summaries = []
            total_channels = 0
            for n, port in enumerate(xdaq.ports):
                connected = [s for s in port if s.chip.num_channels_per_stream() > 0]
                channels = sum(s.chip.num_channels_per_stream() for s in connected)
                total_channels += channels
                if connected:
                    chips = ", ".join(str(s.chip) for s in connected)
                    port_summaries.append((n + 1, channels, chips))

            device_results['headstages'][mode] = xdaq.ports.to_dict()

            if port_summaries:
                hs_table = Table(
                    box=box.SIMPLE_HEAVY,
                    show_header=True,
                    header_style="bold",
                    padding=(0, 1),
                )
                hs_table.add_column("Port", justify="center", style="cyan", width=6)
                hs_table.add_column("Channels", justify="center", style="green", width=10)
                hs_table.add_column("Headstages", style="white")
                for port_num, ch, chips in port_summaries:
                    hs_table.add_row(str(port_num), str(ch), chips)
                console.print(
                    Panel(
                        hs_table,
                        title=f"[bold]{label}[/bold]",
                        subtitle=f"[dim]{total_channels} channels total[/dim]",
                        border_style="green",
                        box=box.ROUNDED,
                    )
                )
            else:
                console.print(f"  [dim]{label}:[/dim] No headstages detected")
    except Exception as e:
        device_results['headstages'][mode] = str(e)
        console.print(f"  [red]{label}: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="XDAQ Self-Diagnosis Tool")
    parser.add_argument('--no-pause', action='store_true', help="Don't wait for keypress at exit")
    parser.add_argument('--json', action='store_true', help="Output raw JSON only (no TUI)")
    args = parser.parse_args()

    if args.json:
        console = Console(stderr=True, highlight=False)
        run_diagnosis(console)
        print(json.dumps(results, indent=2))
    else:
        console = Console(highlight=False)
        run_diagnosis(console)
        console.print()

    report_path = Path('diagnostic_report.json')
    report_path.write_text(json.dumps(results, indent=2))

    if not args.json:
        console.print(f"[dim]Report saved to:[/dim] {report_path.resolve()}")

    if not args.no_pause and not args.json:
        input('\nPress Enter to exit…')


if __name__ == '__main__':
    main()
