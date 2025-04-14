from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union

import numpy as np
from dataclass_wizard import JSONWizard


class Registers:

    def __init__(self, width, length):
        if width > 4:
            raise ValueError('Width above 64 bits not supported')
        self.width = width
        self.width_bits = width * 8
        self.length = length
        self.dtype = np.dtype(f'<u{width}')
        self.registers = np.zeros((length,), dtype=self.dtype)

    def __getitem__(self, key):
        return self.registers[key]

    def __iter__(self):
        return iter(self.registers)

    def setbits(self, reg, value, start, *, end=None, bits=None):
        if bits is None:
            if end is None:
                raise ValueError('Either bits or end must be specified')
            bits = end - start + 1
        if start + bits > self.width_bits:
            raise ValueError('Start + bits exceeds register width')
        if value > (1 << bits) - 1:
            raise ValueError(
                f'Value {value} exceeds {bits} bits when setting bits {start} to {end} of register {reg}'
            )
        mask = ((1 << bits) - 1) << start
        self.registers[reg] = (self.registers[reg]
                               & ~type(self.registers[0])(mask)) | ((value << start) & mask)

    def getbits(self, reg, start, *, end=None, bits=None):
        if bits is None:
            if end is None:
                raise ValueError('Either bits or end must be specified')
            bits = end - start + 1
        if start + bits > self.width_bits:
            raise ValueError('Start + bits exceeds register width')
        mask = ((1 << bits) - 1) << start
        return (self.registers[reg] & mask) >> start


@dataclass
class NamedRegister(JSONWizard):
    name: str
    reg: int
    start: int
    end: int
    default: Union[str, int] = 0
    mode: str = 'rw'
    bits: int = None

    def __post_init__(self):
        if self.bits is None:
            self.bits = self.end - self.start + 1
        if self.mode not in ('rw', 'ro'):
            raise ValueError(f'Invalid mode {self.mode}')
        if isinstance(self.default, str):
            if self.default.startswith('0x'):
                self.default = int(self.default, 16)
            elif self.default.startswith('0b'):
                self.default = int(self.default, 2)
            else:
                self.default = int(self.default)


@dataclass
class RegisterController(JSONWizard):
    width: int
    length: int
    named_registers: List[NamedRegister]

    def __post_init__(self):
        self.registers = Registers(width=self.width, length=self.length)
        self._sanity_check()
        self.named_registers_dict = {nr.name: nr for nr in self.named_registers}
        self._set_default()

    def _sanity_check(self):
        names = [nr.name for nr in self.named_registers]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate names')
        for nr in self.named_registers:
            if nr.reg >= self.registers.length:
                raise ValueError(f'Register {nr.reg} out of range')
            if nr.start >= self.registers.width_bits:
                raise ValueError(f'Start {nr.start} out of range')
            if nr.start + nr.bits > self.registers.width_bits:
                raise ValueError(f'Start {nr.start} + bits {nr.bits} out of range')

    def _set_default(self):
        for nr in self.named_registers:
            if nr.default != 0:
                self.set(nr.name, nr.default)

    def set(self, name, value):
        nr = self.named_registers_dict[name]
        if nr.mode == 'ro':
            raise ValueError(f'Register {name} is read-only')
        self.registers.setbits(nr.reg, value, nr.start, bits=nr.bits)

    def get(self, name):
        nr = self.named_registers_dict[name]
        return self.registers.getbits(nr.reg, nr.start, bits=nr.bits)


class OperandType(Enum):
    ADDRESS = 'ADDRESS'
    REGISTER = 'REGISTER'
    IMMEDIATE = 'IMMEDIATE'
    VARIABLE = 'VARIABLE'


@dataclass
class Operand(JSONWizard):
    bits: int
    type: OperandType
    value: Union[int, str] = None
    name: str = None

    def __post_init__(self):
        if isinstance(self.value, str):
            if self.value.startswith('0x'):
                self.value = int(self.value, 16)
            elif self.value.startswith('0b'):
                self.value = int(self.value, 2)
            else:
                self.value = int(self.value)


@dataclass
class Operation(JSONWizard):
    opcode: int
    operands: List[Operand]


@dataclass
class ISA(JSONWizard):
    operations: Dict[str, Operation]
    opcode_bits: int
    instruction_bits: int
    dtype: Union[str, None] = None

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = np.dtype(f'<u{self.instruction_bits//8}')

    def __iter__(self):
        return iter(self.operations)

    def __getitem__(self, name) -> Operation:
        return self.operations[name]


def machinecode(opname: str, isa: ISA, registers: Registers, **kwargs):
    op = isa[opname]
    shift = isa.instruction_bits - isa.opcode_bits
    binary = int(op.opcode << shift)
    for o in op.operands:
        shift -= o.bits
        if o.type == OperandType.ADDRESS:
            if o.name not in kwargs:
                raise ValueError(f'Address operand {o.name} not specified')
            binary |= int(kwargs[o.name] << shift)
        elif o.type == OperandType.IMMEDIATE:
            binary |= int(o.value << shift)
        elif o.type == OperandType.REGISTER:
            if o.name not in kwargs:
                raise ValueError(f'Register operand {o.name} not specified')
            binary |= int(registers[kwargs[o.name]] << shift)
        elif o.type == OperandType.VARIABLE:
            if o.name not in kwargs:
                raise ValueError(f'Variable operand {o.name} not specified')
            binary |= int(kwargs[o.name] << shift)
        else:
            raise ValueError(f'Invalid operand type {o.type}')

    if binary >= (1 << isa.instruction_bits):
        raise ValueError(
            f'Overflow occurred when encoding {opname} with {kwargs}, {binary:b}>={1 << isa.instruction_bits:b}'
        )
    return binary
