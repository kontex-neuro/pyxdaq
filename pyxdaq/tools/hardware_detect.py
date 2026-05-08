"""
OS-level hardware detection for XDAQ devices.

Detects XDAQ hardware on USB and PCIe buses before driver initialization,
enabling clearer diagnostics ("hardware not found" vs "driver failed").
"""

import json
import platform
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# Known XDAQ hardware identifiers
OPALKELLY_USB_VID = 0x151F  # Gen1 USB (OpalKelly)
XDAQ_PCIE_VID = 0x4B43  # Gen2 PCIe (Thunderbolt)
XDAQ_PCIE_DID = 0x4B58  # Gen2 PCIe device ID


@dataclass
class HardwareDevice:
    bus: str  # "usb" or "pci"
    vendor_id: int
    device_id: int = 0
    serial: str = ""
    name: str = ""
    location: str = ""
    extra: dict = field(default_factory=dict)


def detect_xdaq_hardware() -> dict:
    """Detect XDAQ hardware at the OS level.

    Returns a dict with:
        'gen1': list of Gen1 (USB/OpalKelly) devices found
        'gen2': list of Gen2 (PCIe/Thunderbolt) devices found
        'errors': list of non-fatal errors encountered
    """
    result = {'gen1': [], 'gen2': [], 'errors': []}
    system = platform.system()

    try:
        if system == 'Darwin':
            result['gen1'] = _macos_find_usb(OPALKELLY_USB_VID)
            result['gen2'] = _macos_find_pci(XDAQ_PCIE_VID, XDAQ_PCIE_DID)
        elif system == 'Linux':
            result['gen1'] = _linux_find_usb(OPALKELLY_USB_VID)
            result['gen2'] = _linux_find_pci(XDAQ_PCIE_VID, XDAQ_PCIE_DID)
        elif system == 'Windows':
            result['gen1'] = _windows_find_usb(OPALKELLY_USB_VID)
            result['gen2'] = _windows_find_pci(XDAQ_PCIE_VID, XDAQ_PCIE_DID)
        else:
            result['errors'].append(f"Unsupported OS: {system}")
    except Exception as e:
        result['errors'].append(str(e))

    return result


# ── macOS ────────────────────────────────────────────────────────────────────


def _macos_find_usb(vendor_id: int) -> list[HardwareDevice]:
    try:
        out = subprocess.run(
            ['system_profiler', 'SPUSBDataType', '-json'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(out.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []

    devices = []
    _macos_walk_usb(data.get('SPUSBDataType', []), vendor_id, devices)
    return devices


def _macos_walk_usb(items, vendor_id: int, out: list):
    """Recursively walk the system_profiler USB JSON tree."""
    if isinstance(items, list):
        for item in items:
            _macos_walk_usb(item, vendor_id, out)
    elif isinstance(items, dict):
        vid = items.get('vendor_id', '')
        try:
            if isinstance(vid, str):
                # Format: "0x151f" or "0x151f  (Opal Kelly Incorporated)"
                vid_hex = vid.split()[0] if vid else ''
                vid_int = int(vid_hex, 16) if vid_hex else -1
            else:
                vid_int = int(vid) if vid is not None else -1
        except (ValueError, TypeError):
            vid_int = -1

        if vid_int == vendor_id:
            out.append(
                HardwareDevice(
                    bus='usb',
                    vendor_id=vendor_id,
                    name=items.get('_name', ''),
                    serial=items.get('serial_num', ''),
                    location=items.get('location_id', ''),
                    extra={'manufacturer': items.get('manufacturer', '')},
                )
            )
        # Recurse into all values (nested hubs, _items, etc.)
        for val in items.values():
            if isinstance(val, (dict, list)):
                _macos_walk_usb(val, vendor_id, out)


def _macos_find_pci(vendor_id: int, device_id: int) -> list[HardwareDevice]:
    try:
        out = subprocess.run(
            ['system_profiler', 'SPPCIDataType', '-json'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        data = json.loads(out.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []

    devices = []
    for item in data.get('SPPCIDataType', []):
        if not isinstance(item, dict):
            continue
        vid_str = item.get('sppci_vendor-id', item.get('vendor_id', ''))
        did_str = item.get('sppci_device-id', item.get('device_id', ''))
        try:
            vid = int(vid_str, 16) if isinstance(vid_str, str) else int(vid_str)
            did = int(did_str, 16) if isinstance(did_str, str) else int(did_str)
        except (ValueError, TypeError):
            continue
        if vid == vendor_id and did == device_id:
            devices.append(
                HardwareDevice(
                    bus='pci',
                    vendor_id=vendor_id,
                    device_id=device_id,
                    name=item.get('_name', ''),
                    location=item.get('sppci_slot_name', item.get('slot_name', '')),
                    extra={
                        'link_width': item.get('sppci_link-width', ''),
                        'link_speed': item.get('sppci_link-speed', ''),
                        'driver': item.get('sppci_driver_installed', ''),
                    },
                )
            )
    return devices


# ── Linux ────────────────────────────────────────────────────────────────────


def _linux_find_usb(vendor_id: int) -> list[HardwareDevice]:
    devices = []
    sysfs = Path('/sys/bus/usb/devices')
    if not sysfs.exists():
        return devices
    vid_hex = f'{vendor_id:04x}'
    for dev_path in sysfs.iterdir():
        id_vendor = dev_path / 'idVendor'
        if not id_vendor.exists():
            continue
        try:
            if id_vendor.read_text().strip() == vid_hex:
                serial = ''
                serial_path = dev_path / 'serial'
                if serial_path.exists():
                    serial = serial_path.read_text().strip()
                product = ''
                product_path = dev_path / 'product'
                if product_path.exists():
                    product = product_path.read_text().strip()
                pid_str = ''
                pid_path = dev_path / 'idProduct'
                if pid_path.exists():
                    pid_str = pid_path.read_text().strip()
                devices.append(
                    HardwareDevice(
                        bus='usb',
                        vendor_id=vendor_id,
                        device_id=int(pid_str, 16) if pid_str else 0,
                        serial=serial,
                        name=product,
                        location=dev_path.name,
                    )
                )
        except (OSError, ValueError):
            continue
    return devices


def _linux_find_pci(vendor_id: int, device_id: int) -> list[HardwareDevice]:
    devices = []
    sysfs = Path('/sys/bus/pci/devices')
    if not sysfs.exists():
        return devices
    vid_hex = f'0x{vendor_id:04x}'
    did_hex = f'0x{device_id:04x}'
    for dev_path in sysfs.iterdir():
        try:
            vid_file = dev_path / 'vendor'
            did_file = dev_path / 'device'
            if not vid_file.exists() or not did_file.exists():
                continue
            if (vid_file.read_text().strip() == vid_hex
                    and did_file.read_text().strip() == did_hex):
                devices.append(
                    HardwareDevice(
                        bus='pci',
                        vendor_id=vendor_id,
                        device_id=device_id,
                        location=dev_path.name,
                    )
                )
        except OSError:
            continue
    return devices


# ── Windows ──────────────────────────────────────────────────────────────────


def _windows_find_usb(vendor_id: int) -> list[HardwareDevice]:
    vid_str = f'VID_{vendor_id:04X}'
    try:
        out = subprocess.run(
            [
                'powershell', '-NoProfile', '-Command',
                f"Get-PnpDevice -Class USB | Where-Object {{ $_.HardwareID -match '{vid_str}' }} | "
                "Select-Object FriendlyName, InstanceId | ConvertTo-Json"
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if not out.stdout.strip():
            return []
        data = json.loads(out.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []

    if isinstance(data, dict):
        data = [data]
    devices = []
    for item in data:
        devices.append(
            HardwareDevice(
                bus='usb',
                vendor_id=vendor_id,
                name=item.get('FriendlyName', ''),
                location=item.get('InstanceId', ''),
            )
        )
    return devices


def _windows_find_pci(vendor_id: int, device_id: int) -> list[HardwareDevice]:
    vid_str = f'VEN_{vendor_id:04X}'
    did_str = f'DEV_{device_id:04X}'
    try:
        out = subprocess.run(
            [
                'powershell', '-NoProfile', '-Command',
                f"Get-PnpDevice | Where-Object {{ $_.HardwareID -match '{vid_str}' -and $_.HardwareID -match '{did_str}' }} | "
                "Select-Object FriendlyName, InstanceId | ConvertTo-Json"
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if not out.stdout.strip():
            return []
        data = json.loads(out.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []

    if isinstance(data, dict):
        data = [data]
    devices = []
    for item in data:
        devices.append(
            HardwareDevice(
                bus='pci',
                vendor_id=vendor_id,
                device_id=device_id,
                name=item.get('FriendlyName', ''),
                location=item.get('InstanceId', ''),
            )
        )
    return devices
