def const(x):
    return x

#
# Spudnik Protocol
# 
#   The protocol works by sending a string of simple packets which each have a name, a mode, a string of
# parameters, and a checksum. There are three modes defined Read, Write, and Execute. The Read mode is
# for requesting data from the connected device, the Write mode is for setting a value in the connected
# device or returning requested data, and the Execute mode is for executing the requested command. The packets
# are encoded as an ASCII text string where the name is any string of alphabetical characters or underscores without
# whitespace characters or other punctuation. The parameters are numbers written as either decimal integers or decimal
# floating point. Non-numeric parameters are not supported. The 8-bit checksum is encoded as a decimal integer between 0 
# and 255. The beginning of a packet is marked with a left square-bracket and the end of a command is marked with the
# right square bracket. During parsing, whitespace characters between parameters are ignored.
#
#   Basic Command Structure - [<mode><name>,(<param>,)*<chk>]
#   
#   Mode Marking Prefix:
#       Read - ?
#       Write - =
#       Execute - #
#
#   The checksum is calculated as the lower 8-bits of the sum of the numeric values of all the characters from the first
# character of the name to the comma which preceeds the checksum characters (the left and right brackets are excluded, 
# as are the actual checksum characters).
#
#
# Processor Callbacks:
#   The processor allows for callbacks to be registered for each of the possible endpoint combinations of mode and name.
# Callback functions are of the general form: func(name, mode, *parameters). In this function, the first argument is the
# name of the packet/command. While the second argument is the operation mode and the final arguments are some number of
# numeric parameters. The callbacks are not required to be variadic functions however. If it is known that a callback will
# only ever be called with a fixed number of parameters, it is entirely allowable to use a fixed signature callback function.
# It should be noted, however, that calling such a fuction with an improper number of parameters will result in an exception
# being raised which the protocol processor will not catch.
#

class SpudnikProtocolProcessor:
    ReadMode = const(0)
    WriteMode = const(1)
    ExecMode = const(2)

    def __init__(self):
        self.__exceptions = False
        self.__buffer = bytes([])
        self.__callbacks = {}

    def __call__(self, data):
        exceptions = self.__exceptions
        callbacks = self.__callbacks
        if isinstance(data, str):
            data = data.encode()
        self.__buffer += data
        done = False
        results = []
        while not done:
            self.__drop_junk()
            end = self.__cmd_end()
            if end >= 0:
                cmd = self.__buffer[:end+1]
                self.__buffer = self.__buffer[end+1:]
                result = self.__parse(cmd)
                # print(results)
                if result is not None:
                    results.append(result)
                    name, mode, params = result
                    if (name, mode) in callbacks.keys():
                        callbacks[(name, mode)](name, mode, *params)
            else:
                done = True
        return results

    def register_callback(self, name, callback, mode=None):
        if not name in self.__callbacks.keys():
            if mode is None:
                self.__callbacks[(name, SpudnikProtocolProcessor.ReadMode)] = callback
                self.__callbacks[(name, SpudnikProtocolProcessor.WriteMode)] = callback
                self.__callbacks[(name, SpudnikProtocolProcessor.ExecMode)] = callback
            else:
                self.__callbacks[(name, mode)] = callback
            return True
        return False

#    def register_callback(self, callback, mode=None):
#        self.register_callback(callback.__name__, callback, mode=mode)

    def encode_read(self, name, *params):
        return self.__encode_packet(name, SpudnikProtocolProcessor.ReadMode, *params)

    def encode_write(self, name, *params):
        return self.__encode_packet(name, SpudnikProtocolProcessor.WriteMode, *params)

    def encode_exec(self, name, *params):
        return self.__encode_packet(name, SpudnikProtocolProcessor.ExecMode, *params)

    def __calc_chk(self, buffer):
        chk = 0
        for c in buffer:
            chk += ord(c)
        return chk & 0xFF

    def __encode_packet(self, name, mode, *params):
        mode_char = '#'
        if mode == SpudnikProtocolProcessor.ReadMode:
            mode_char = '?'
        elif mode == SpudnikProtocolProcessor.WriteMode:
            mode_char = '='

        # print('params = {}'.format(params))
        param_string = '' if len(params) == 0 else ' '.join(['{},'.format(v) for v in params])
        # print('param_string = {}'.format(param_string))
        protected_string = '{}{},{}{}'.format(mode_char, name, '' if len(param_string) == 0 else ' ', param_string)
        # print('protected_string = {}'.format(protected_string))
        chk_string = '{:d}'.format(self.__calc_chk(protected_string))
        # print('chk_string = {}'.format(chk_string))
        return '[{} {}]'.format(protected_string, chk_string) 

    def __drop_junk(self):
        good_idx = self.__buffer.find('['.encode())
        if good_idx < 0:
            self.__buffer = bytes()
        elif good_idx > 0: 
            self.__buffer = self.__buffer[good_idx:]

    def __cmd_end(self):
        return self.__buffer.find(']'.encode())

    def __parse(self, cmd):
        exceptions = self.__exceptions
        cmd = cmd.decode()
        # print('cmd = {}'.format(cmd))
        if (not cmd[0] == '[') or (not cmd[-1] == ']'): # Improper Command Wrapping
            return None
        packet = cmd[1:-1]
        if not packet.find('[') == -1: # Repeated Start Character
            if exceptions:
                raise Exception('Repeated Start Error -> {}'.format(packet))
            # print('Repeated Start Error -> {}'.format(packet))
            return None
        args = [v.strip() for v in packet.split(',')]
        # print('args = {}'.format(args))
        protected_string = cmd[1:cmd.rfind(',')+1]
        # print('protected string -> {}'.format(protected_string))
        chk = self.__calc_chk(protected_string)
        if not int(args[-1]) == chk:
            if exceptions:
                raise Exception('Checksum Mismatch -> {} != {}'.format(chk, args[-1]))
            return None
        mode_char = args[0][0]
        mode = None
        if mode_char == '?':
            mode = SpudnikProtocolProcessor.ReadMode
        elif mode_char == '=':
            mode = SpudnikProtocolProcessor.WriteMode
        elif mode_char == '#':
            mode = SpudnikProtocolProcessor.ExecMode
        if mode is None:
            if exceptions:
                raise Exception('Invalid Mode Prefix -> {}'.format(mode_char))
            return None
        name = args[0][1:]
        params = []
        for arg in args[1:-1]:
            param = None
            try:
                param = int(arg)
            except Exception:
                pass

            if param is None:
                try:
                    param = float(arg)
                except Exception:
                    if exceptions:
                        raise Exception('Invalid Parameter -> {}'.format(arg))
                    return None
            params.append(param)
        return name, mode, params


if __name__ == '__main__':
    def chunkstring(s, l):
        return [s[0+i:l+i] for i in range(0, len(s), l)]

    def echo_callback(name, mode, *args):
        print('{}[{}] {}'.format(name, mode, args))

    # The test packets are arranged as:
    #   4 correct
    #   1 error - Incorrect CHK
    #   1 error - No packet start
    #   1 error - No packet end
    #   3 correct
    test_packets = '[?servo_zero, 1, 54][#set_servo_zero, 1, 3, 68][=h0, 42, 179][?dt, 67][?dt, 81]=h0, 42, 179][?servo_zero, 1, 54[][?servo_zero, 1, 54]  [=dt, 86, 251],[#dt, 1, 2, 3, 161]'
    protocol = SpudnikProtocolProcessor()
    protocol.register_callback('h0', echo_callback)
    protocol.register_callback('dt', echo_callback)

    for chunk in chunkstring(test_packets, 3):
        try:
            results = protocol(chunk)
            for result in results:
                print(result)
        except Exception as e:
            print(e)

    # print(protocol.encode_write('dt', 86))
    # print(protocol.encode_exec('dt', 1, 2, 3))
    # print(protocol.encode_read('dt'))
    # print(protocol.encode_write('h0', 42))
    # print(protocol.encode_exec('set_servo_zero', 1, 3))
    # print(protocol.encode_read('servo_zero', 1))
