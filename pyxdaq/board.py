import re
from pathlib import Path
from typing import Tuple, Union

from . import constants, ok
from .utils import DebugWrapper


class Board:
    """
    Abstract class for interacting with the okFrontPanel API.
    """

    def __init__(self, debug: bool = False):
        self.dev: ok.okCFrontPanel = None
        self.info: ok.okTDeviceInfo = None
        self.rhs: bool = None
        self.ep: Union[constants.RHD, constants.RHS, None] = None
        self.expander: bool = None
        self.debug = debug
        self.max_datastreams = -1
        self.channels_per_datastream = -1
        dev, self.info = self._get_device()
        self.dev = DebugWrapper(dev) if self.debug else dev

    def GetWireOutValue(self, addr: constants.EndPoints, update: bool = True) -> int:
        if update:
            self.dev.UpdateWireOuts()
        return self.dev.GetWireOutValue(addr.value)

    def SetWireInValue(
        self, addr: constants.EndPoints, value: int, mask: int = 0xffffffff, update: bool = True
    ):
        self.dev.SetWireInValue(addr.value, value, mask)
        if update:
            self.dev.UpdateWireIns()

    def ActivateTriggerIn(self, addr: constants.EndPoints, value: int):
        self.dev.ActivateTriggerIn(addr.value, value)

    def WriteToBlockPipeIn(self, epAddr: constants.EndPoints, blockSize: int, data: bytearray):
        return self.dev.WriteToBlockPipeIn(epAddr.value, blockSize, data)

    def ReadFromBlockPipeOut(self, epAddr: constants.EndPoints, blockSize: int, data: bytearray):
        ret = self.dev.ReadFromBlockPipeOut(epAddr.value, blockSize, data)
        if ret < 0:
            raise Exception(f'ReadFromBlockPipeOut failed with error code {ret}')
        return ret

    def SendMultiUse(self, trig: constants.EndPoints, bit: int, value: int, mask: int = 0xffffffff):
        self.dev.SetWireInValue(self.ep.WireInMultiUse.value, value, mask)
        self.dev.UpdateWireIns()
        self.dev.ActivateTriggerIn(trig.value, bit)

    @classmethod
    def _get_device(cls) -> Tuple[ok.okCFrontPanel, ok.okTDeviceInfo]:
        ok.GetAPIVersionString()
        dev = ok.okCFrontPanel()
        supported = [ok.okPRODUCT_XEM7310A75, ok.okPRODUCT_XEM6010LX45, ok.okPRODUCT_XEM6310LX45]
        for i in range(dev.GetDeviceCount()):
            md, sn = dev.GetDeviceListModel(i), dev.GetDeviceListSerial(i)
            if md in supported:
                res = dev.OpenBySerial(sn)
                if res != ok.okCFrontPanel.NoError:
                    print(f'Open failed {res}')
                    continue
                info = ok.okTDeviceInfo()
                dev.GetDeviceInfo(info)
                return dev, info
        raise RuntimeError('No supported device found')

    @classmethod
    def _get_bitfilename(cls, info: ok.okTDeviceInfo, rhs: bool = False) -> str:
        return re.sub(
            r'xem([^-]+)-([^-]+)', r'x{}r\1\2.bit'.format('s' if rhs else ''),
            info.productName.lower()
        )

    def detect_expander(self):
        raise NotImplementedError

    def config_fpga(self, rhs: bool = False, bitfile: str = None) -> Tuple[int, int]:
        if bitfile is None:
            bitfile = self._get_bitfilename(self.info, rhs)
        if not Path(bitfile).exists():
            raise FileNotFoundError(f'bitfile {bitfile} not found')
        error_code = self.dev.ConfigureFPGA(bitfile)
        if error_code != ok.okCFrontPanel.NoError:
            raise RuntimeError(f'Configure FPGA failed {error_code}')
        if not self.dev.IsFrontPanelEnabled():
            raise RuntimeError('FrontPanel not enabled')
        self.ep = constants.RHS if rhs else constants.RHD
        self.rhs = rhs
        self.max_datastreams = 8 if rhs else 32
        self.channels_per_datastream = 16 if rhs else 32
        boardId = self.GetWireOutValue(self.ep.WireOutBoardId)
        boardVersion = self.GetWireOutValue(self.ep.WireOutBoardVersion, False)
        self.reset_board()
        self.expander = self.detect_expander()
        return boardId, boardVersion

    def reset_board(self):
        """
        This clears all auxiliary command RAM banks, clears the USB FIFO, and resets the
        per-channel sampling rate to 30.0 kS/s/ch.
        """
        self.SetWireInValue(self.ep.WireInResetRun, 1, 1)
        self.SetWireInValue(self.ep.WireInResetRun, 0, 1)

    def reset_fpga(self):
        """
        Low level function to reset the FPGA.
        """
        self.dev.ResetFPGA()
