import struct


class FCCommand:
    Unlock = 0x01
    SetSystemMode = 0x02
    SetControlMode = 0x03
    WriteData = 0x05
    RequestData = 0x06
    HILUpdate = 0x0C
    HILSync = 0x0D

    __cmd_list = [
        Unlock,
        SetSystemMode,
        SetControlMode,
        WriteData,
        RequestData,
        HILUpdate,
        HILSync
    ]

    @classmethod
    def valid(cls, cmd):
        return cmd in cls.__cmd_list


class FCSystemMode:
    Normal = 0x01
    Calibration = 0x02
    Datalog = 0x04

    @classmethod
    def valid(cls, mode):
        return cls.Normal <= mode <= cls.Datalog


class FCControlMode:
    Normal = 0x01
    HIL = 0x02

    @classmethod
    def valid(cls, mode):
        return cls.Normal <= mode <= cls.Datalog


def crc16_ccitt(data: bytes):
    #
    # This is based on code found at:
    #   https://stackoverflow.com/questions/25239423/crc-ccitt-16-bit-python-manual-calculation

    crc = 0xFFFF
    msb = crc >> 8
    lsb = crc & 255
    for c in data:
        x = c ^ msb
        x ^= (x >> 4)
        msb = (lsb ^ (x >> 3) ^ (x << 4)) & 255
        lsb = (x ^ (x << 5)) & 255
    return (msb << 8) + lsb


class FCPacket:
    def __init__(self, cmd, idx=0x00, data=None, is_float=True):
        self.__cmd = cmd
        self.__idx = idx
        self.__data = data
        self.__is_float = is_float

    def __str__(self):
        return '{}()'.format(self.__class__.__name__)

    @property
    def valid(self):
        return False

    @property
    def cmd(self):
        return self.__cmd

    @property
    def idx(self):
        return self.__idx

    @property
    def data(self):
        return self.__data

    @property
    def databytes(self):
        if self.__data is None:
            return [0x00] * 4
        elif self.__is_float:
            return list(struct.pack('f', self.__data))
        else:
            return list(self.__data.to_bytes(4, byteorder='big'))

    @property
    def is_float(self):
        return self.__is_float

    def encode(self):
        data = [self.__cmd, self.idx] + self.databytes
        crc = crc16_ccitt(data)
        # print('0x{:X}'.format(crc))
        crc_h = (crc >> 8) & 0xFF
        crc_l = crc & 0xFF
        # print('CRCH = 0x{:X}, CRCL = 0x{:X}'.format(crc_h, crc_l))
        data += [crc_h, crc_l]
        # print('data = {}'.format(data))
        return bytes(data)


class UnlockPacket(FCPacket):
    def __init__(self, unlock_command=None):
        unlock_command = 0xFF if unlock_command is None else unlock_command
        FCPacket.__init__(self, FCCommand.Unlock, 0x86, unlock_command, is_float=False)

    def __str__(self):
        cmd = self.unlock_command
        cmd_str = 'None' if cmd is None else '0x{:02X}'.format(cmd)
        return '{}(cmd = {})'.format(self.__class__.__name__, cmd_str)

    @property
    def valid(self):
        return False

    @property
    def unlock_command(self):
        return None if self.data is None else self.data[-1]

    @unlock_command.setter
    def unlock_command(self, cmd):
        # TODO: VALIDATE UNLOCK COMMANDS
        self.data(cmd)
        return self.data


class SystemModePacket(FCPacket):
    def __init__(self, mode=None):
        mode = 0xFF if mode is None else mode
        FCPacket.__init__(self, FCCommand.SetSystemMode, 0x42, mode, is_float=False)

    @property
    def valid(self):
        return FCSystemMode.valid(self.mode)

    @property
    def mode(self):
        return self.data

    @mode.setter
    def mode(self, _mode):
        if FCSystemMode.valid(_mode):
            self.data(_mode)
        return self.data


class ControlModePacket(FCPacket):
    def __init__(self, mode=None):
        mode = 0xFF if mode is None else mode
        FCPacket.__init__(self, FCCommand.SetControlMode, 0x42, mode, is_float=False)

    @property
    def valid(self):
        return FCControlMode.valid(self.mode)

    @property
    def mode(self):
        return self.data

    @mode.setter
    def mode(self, _mode):
        if FCSystemMode.valid(_mode):
            self.data(_mode)
        return self.data


class WriteDataPacket(FCPacket):
    def __init__(self, idx=None, data=None, is_float=True):
        idx = 0xFF if idx is None else idx
        data = (0.0 if is_float else 0x00) if data is None else data
        FCPacket.__init__(self, FCCommand.WriteData, idx, data, is_float=is_float)

    @property
    def valid(self):
        return True  # TODO: FIGURE OUT VALIDATION...


class RequestDataPacket(FCPacket):
    def __init__(self, idx=None):
        idx = 0xFF if idx is None else idx
        FCPacket.__init__(self, FCCommand.RequestData, idx, 0x00, is_float=False)

    @property
    def valid(self):
        return FCControlMode.valid(self.mode)


class HILUpdatePacket(FCPacket):
    def __init__(self, dt=None):
        FCPacket.__init__(self, FCCommand.HILUpdate, 0x42, 1 if dt is None else int(dt), is_float=False)

    @property
    def valid(self):
        return (self.data is not None) and (self.data > 0)

    @property
    def dt(self):
        return self.data

    @dt.setter
    def dt(self, _dt):
        if _dt > 0:
            self.data(_dt)
        return self.data


class HILSyncPacket(FCPacket):
    def __init__(self):
        FCPacket.__init__(self, FCCommand.HILSync)


class FCPacketParser:
    def __init__(self):
        self.__buffer = []

    @property
    def buffer(self):
        return self.__buffer

    def __call__(self, data):
        packets = []
        self.__buffer += list(data)
        while len(self.__buffer) >= 8:
            crc_calc = crc16_ccitt(self.__buffer[:6])
            crc_tgt = int.from_bytes(self.__buffer[6:8], byteorder='big')
            # print('CRC Calc - 0x{:X}'.format(crc_calc))
            # print('CRC Tgt - 0x{:X}'.format(crc_tgt))
            if crc_tgt == crc_calc:
                # print('Detected a valid CRC')
                packet_bytes = self.__buffer[:8]
                self.__buffer = self.__buffer[8:]  # Drop a whole packet
                cmd = packet_bytes[0]
                idx = packet_bytes[1]
                data = packet_bytes[2:6]
                packet = None
                if cmd == FCCommand.Unlock:
                    packet = UnlockPacket(unlock_command=data)
                elif cmd == FCCommand.RequestData:
                    packet = RequestDataPacket(idx=idx)
                elif cmd == FCCommand.WriteData:
                    is_float = True  # TODO: THIS NEEDS TO BE BASED ON THE INDEX AND THE DEVICE CONNECTED TO THE SYSTEM
                    packet = WriteDataPacket(idx=idx, data=data, is_float=is_float)
                elif cmd == FCCommand.HILUpdate:
                    packet = HILUpdatePacket(dt=struct.unpack('f', bytearray(data)))
                elif cmd == FCCommand.HILSync:
                    packet = HILSyncPacket()

                if packet is not None:
                    packets.append(packet)
                else:
                    print('Failed to parse the packet with - {}'.format(packet_bytes))
            else:
                self.__buffer = self.__buffer[1:]  # Drop one byte
        return packets


if __name__ == '__main__':
    parser = FCPacketParser()
    unlock_system_mode = UnlockPacket(FCCommand.SetSystemMode)
    set_system_mode = SystemModePacket(FCSystemMode.Normal)
    unlock_write = UnlockPacket(FCCommand.WriteData)
    hil_update = HILUpdatePacket(20000)
    input_packets = [unlock_system_mode, unlock_write, hil_update, 0x04, 0x03, unlock_write]
    packet_bytes = []
    for packet in input_packets:
        if isinstance(packet, FCPacket):
            packet_bytes += packet.encode()
        else:
            packet_bytes.append(packet)
    packets = parser(packet_bytes)
    print([str(p) for p in packets])

    print()

    packet_bytes = set_system_mode.encode()
    print(['0x{:X}'.format(b) for b in packet_bytes])
    crc = crc16_ccitt(bytes([0x02, 0x42, 0x00, 0x00, 0x00, 0x01]))
    print('0x{:X}'.format(crc))
