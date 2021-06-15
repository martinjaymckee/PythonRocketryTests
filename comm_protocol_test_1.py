#
# Standard Library
#
import signal
import sys
import time


#
# Import 3rd Party Libraries
#
import serial
import serial.tools.list_ports


#
# Import Project Libraries
#
import comm_protocol


t_inter_char = 0.00025


def send(comm, packet, debug=False):
    packet_bytes = packet.encode()
    for b in packet_bytes:
        comm.write(bytes([b]))
        time.sleep(t_inter_char)  # NOTE: THIS IS A HACK TO MAKE THIS WORK WITH THE LPCXPRESSO845 BOARD
    if debug:
        print('[{}]'.format(', '.join(['0x{:02X}'.format(b) for b in packet_bytes])))


class PacketProcessor:
    def __init__(self, comm):
        self.__comm = comm
        self.__column = 0
        self.__count = 0

    def __call__(self, packet):
        if isinstance(packet, comm_protocol.HILSyncPacket):
            print('*')
            send(self.__comm, comm_protocol.HILUpdatePacket(20000))
        else:
            print(packet)


def enterHILMode(comm):
    send(comm, comm_protocol.UnlockPacket(comm_protocol.FCCommand.SetSystemMode))
    send(comm, comm_protocol.SystemModePacket(comm_protocol.FCSystemMode.Normal))
    send(comm, comm_protocol.UnlockPacket(comm_protocol.FCCommand.SetControlMode))
    send(comm, comm_protocol.ControlModePacket(comm_protocol.FCControlMode.HIL))
    send(comm, comm_protocol.HILUpdatePacket(20000))


def exitHILMode(comm):
    send(comm, comm_protocol.UnlockPacket(comm_protocol.FCCommand.SetControlMode))
    send(comm, comm_protocol.ControlModePacket(comm_protocol.FCControlMode.Normal))
    send(comm, comm_protocol.UnlockPacket(comm_protocol.FCCommand.SetSystemMode))
    send(comm, comm_protocol.SystemModePacket(comm_protocol.FCSystemMode.Normal))


if __name__ == '__main__':
    running = True
    for info in serial.tools.list_ports.comports():
        print(info)

    port = 'COM7'
    baud = 230400
    try:
        print('Connecting to Serial Port - {}'.format(port))
        comm = serial.Serial(port, baudrate=baud)
    except Exception as e:
        print(e)
        quit(-1)

    def signal_handler(*args):
        global running
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    parser = comm_protocol.FCPacketParser()
    processor = PacketProcessor(comm)

    print('Entering HIL Mode')
    enterHILMode(comm)

    try:
        while running:
            if comm.in_waiting > 0:
                packet_bytes = bytes(comm.read(comm.in_waiting))
                print(packet_bytes)
                packets = parser(packet_bytes)
                for packet in packets:
                    processor(packet)
            time.sleep(t_inter_char)
            send(comm, comm_protocol.HILUpdatePacket(20000))
    finally:
        print('\nExiting HIL Mode...')
        exitHILMode(comm)
        print('Shutting Down...')
        while comm.out_waiting > 0:
            time.sleep(0.01)
        comm.close()
        print('Goodbye!')

    # unlock_system_mode = comm_protocol.UnlockPacket(comm_protocol.FCCommand.SetSystemMode)
    # set_system_mode = comm_protocol.SystemModePacket(comm_protocol.FCSystemMode.Normal)
    # unlock_write = comm_protocol.UnlockPacket(comm_protocol.FCCommand.WriteData)
    # hil_update = comm_protocol.HILUpdatePacket(1/100)
    # input_packets = [unlock_system_mode, unlock_write, hil_update, 0x04, 0x03, unlock_write]
    # packet_bytes = []
    # for packet in input_packets:
    #     if isinstance(packet, comm_protocol.FCPacket):
    #         packet_bytes += packet.encode()
    #     else:
    #         packet_bytes.append(packet)
    # packets = parser(packet_bytes)
    # print([str(p) for p in packets])
    #
    # print()
    #
    # packet_bytes = set_system_mode.encode()
    # print(['0x{:X}'.format(b) for b in packet_bytes])
    # crc = comm_protocol.crc16_ccitt(bytes([0x02, 0x42, 0x00, 0x00, 0x00, 0x01]))
    # print('0x{:X}'.format(crc))
