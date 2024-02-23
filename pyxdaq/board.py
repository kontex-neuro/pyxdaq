from pathlib import Path
from typing import Callable, Tuple, Union

from . import ok
from .constants import EndPoints
from .utils import DebugWrapper


class Board:

    def __init__(self, debug: Union[bool, Callable] = False):
        raise NotImplementedError

    def is_open(self) -> bool:
        raise NotImplementedError

    def GetWireOutValue(self, addr: EndPoints, update: bool = True) -> int:
        raise NotImplementedError

    def SetWireInValue(
        self, addr: EndPoints, value: int, mask: int = 0xffffffff, update: bool = True
    ):
        raise NotImplementedError

    def ActivateTriggerIn(self, addr: EndPoints, value: int):
        raise NotImplementedError

    def WriteToBlockPipeIn(self, epAddr: EndPoints, blockSize: int, data: bytearray):
        raise NotImplementedError

    def ReadFromBlockPipeOut(self, epAddr: EndPoints, blockSize: int, data: bytearray):
        raise NotImplementedError

    def SendTrig(
        self, trig: EndPoints, bit: int, epAddr: EndPoints, value: int, mask: int = 0xffffffff
    ):
        raise NotImplementedError

    def config_fpga(self, bitfile: str = None) -> Tuple[int, int]:
        raise NotImplementedError


class OkBoard(Board):
    """
    Abstract class for interacting with the okFrontPanel API.
    """

    def __init__(self, debug: Union[bool, Callable] = False, dev: ok.okCFrontPanel = None):
        if dev is None:
            dev = self._get_device()
        self.dev = DebugWrapper(dev, debug) if debug else dev

    def is_open(self) -> bool:
        return self.dev.IsOpen()

    def GetWireOutValue(self, addr: EndPoints, update: bool = True) -> int:
        if update:
            self.dev.UpdateWireOuts()
        return self.dev.GetWireOutValue(addr.value)

    def SetWireInValue(
        self, addr: EndPoints, value: int, mask: int = 0xffffffff, update: bool = True
    ):
        self.dev.SetWireInValue(addr.value, value, mask)
        if update:
            self.dev.UpdateWireIns()

    def ActivateTriggerIn(self, addr: EndPoints, value: int):
        self.dev.ActivateTriggerIn(addr.value, value)

    def WriteToBlockPipeIn(self, epAddr: EndPoints, blockSize: int, data: bytearray):
        ret = self.dev.WriteToBlockPipeIn(epAddr.value, blockSize, data)
        if ret < 0:
            raise RuntimeError(f'WriteToBlockPipeIn failed with error code {ret}')
        return ret

    def ReadFromBlockPipeOut(self, epAddr: EndPoints, blockSize: int, data: bytearray):
        ret = self.dev.ReadFromBlockPipeOut(epAddr.value, blockSize, data)
        if ret < 0:
            raise RuntimeError(f'ReadFromBlockPipeOut failed with error code {ret}')
        return ret

    def SendTrig(
        self, trig: EndPoints, bit: int, epAddr: EndPoints, value: int, mask: int = 0xffffffff
    ):
        self.dev.SetWireInValue(epAddr.value, value, mask)
        self.dev.UpdateWireIns()
        self.dev.ActivateTriggerIn(trig.value, bit)

    @classmethod
    def _get_device(cls) -> ok.okCFrontPanel:
        dev = ok.okCFrontPanel()
        supported = [ok.okPRODUCT_XEM7310A75, ok.okPRODUCT_XEM6310LX45]
        for i in range(dev.GetDeviceCount()):
            md, sn = dev.GetDeviceListModel(i), dev.GetDeviceListSerial(i)
            if md in supported:
                res = dev.OpenBySerial(sn)
                if res != ok.okCFrontPanel.NoError:
                    print(f'Open failed {res}')
                    continue
                return dev
        raise RuntimeError('No supported device found')

    def config_fpga(self, bitfile: Union[str, Path]) -> Tuple[int, int]:
        if not Path(bitfile).exists():
            raise FileNotFoundError(f'bitfile {bitfile} not found')
        error_code = self.dev.ConfigureFPGA(str(bitfile))
        if error_code != ok.okCFrontPanel.NoError:
            raise RuntimeError(f'Configure FPGA failed {error_code}')
        if not self.dev.IsFrontPanelEnabled():
            raise RuntimeError('FrontPanel not enabled')
