import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtSerialPort as QtSerialPort

class SerialInfo:
    default_bauds = [9600, 19200, 38400, 57600, 115200, 230400]


class SerialControl(QtWidgets.QWidget):
    connected = QtCore.pyqtSignal(QtSerialPort.QSerialPortInfo, QtSerialPort.QSerialPort)
    disconnected = QtCore.pyqtSignal(QtSerialPort.QSerialPortInfo)
    ports_changed = QtCore.pyqtSignal(set, set)
    
    def __init__(self, parent=None, auto_search=True):
        QtWidgets.QWidget.__init__(self, parent=parent)
        layout = QtWidgets.QGridLayout()
        self.__port_list = QtWidgets.QListWidget()
        self.__ports = {}
        layout.addWidget(self.__port_list, 0, 0, 3, 3)
        self.__baud_list = QtWidgets.QComboBox()
        self.__baud_list.addItems([str(b) for b in SerialInfo.default_bauds])
        self.__baud_list.setCurrentIndex(5)
        self.__baud_label = QtWidgets.QLabel('Baud')
        self.__baud_units = QtWidgets.QLabel('bps')
        layout.addWidget(self.__baud_label, 3, 0, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(self.__baud_list, 3, 1)
        layout.addWidget(self.__baud_units, 3, 2)
        self.__connect_button = QtWidgets.QPushButton('Connect')
        self.__connect_button.setStyleSheet('QPushButton { background-color: green }')        
        self.__connect_button.clicked.connect(self.__process_connection)
        layout.addWidget(self.__connect_button,4, 0, 1, 3)
        self.setLayout(layout)        
        self.__ser = None
        self.__ser_info = None
        self.__search_for_ports()
        self.__auto_search = auto_search
        self.__auto_search_time_ms = 1500
        self.__auto_search_timer = QtCore.QTimer(self)
        self.__auto_search_timer.timeout.connect(self.__search_for_ports)
        
        if self.__auto_search:
            self.__auto_search_timer.start(self.__auto_search_time_ms)
        
    def __del__(self):
        if self.__ser is not None:
            self.__ser.close()
            self.__ser = None
            
        if self.__auto_search_timer.isActive():
            self.__auto_search_timer.stop()
            del self.__auto_search_timer

    @property
    def autosearch(self):
        return self.__auto_search
    
    @autosearch.setter
    def autosearch(self, _enabled):
        self.__auto_search = _enabled
        if _enabled:
            if not self.__auto_search_timer.isActive():
                self.__auto_search_timer.start(self.__auto_search_time_ms)
        else:
            self.__auto_search_timer.stop()
        return self.autosearch

    @property
    def available_ports(self):
        return self.__search_for_ports()
    
    @property
    def connection_active(self):
        return self.__ser is not None
    
    @QtCore.pyqtSlot()    
    def __search_for_ports(self):
        selected = self.__port_list.currentItem()
        if selected is not None:
            selected = selected.text()
        old_ports = set(self.__ports.keys())
        self.__ports = {}
        self.__port_list.clear()
        for port in QtSerialPort.QSerialPortInfo.availablePorts():
            label = port.systemLocation()
            item = QtWidgets.QListWidgetItem(label)
            item.setToolTip('Manufacturer: {}\nDescription: {}'.format(port.manufacturer(), port.description()))
            self.__port_list.addItem(item)
            self.__ports[label] = port
        new_ports = set(self.__ports.keys())

        num_ports = self.__port_list.count()        
        item_selected = False
        if selected is not None:
            items = self.__port_list.findItems(selected, QtCore.Qt.MatchExactly)
            if len(items) > 0:
                self.__port_list.setCurrentItem(items[0])
                item_selected = True
        if (not item_selected) and (num_ports > 0):
            self.__port_list.setCurrentRow(0)

        added_ports = new_ports - old_ports
        removed_ports = old_ports - new_ports
        if (len(added_ports) > 0) or (len(removed_ports) > 0):
            self.ports_changed.emit(added_ports, removed_ports)
        return num_ports

    @QtCore.pyqtSlot()
    def __process_connection(self):
        self.__search_for_ports()
        if self.__ser is None:
            item = self.__port_list.currentItem()
            if item is None:
                print('No port selected!')
                return
            port = item.text()
            if port is not None:
                baud = int(self.__baud_list.currentText())
                ser = QtSerialPort.QSerialPort(port)
                ser.setBaudRate(baud)
                ser.open(QtCore.QIODevice.ReadWrite)
                if ser.isOpen():
                    info = self.__ports[port]
                    self.__connect_button.setText('Disconnect') 
                    self.__connect_button.setStyleSheet('QPushButton { background-color: red }')                    
                    self.__port_list.setEnabled(False)
                    self.__baud_list.setEnabled(False)
                    self.__ser = ser
                    self.__ser_info = info
                    self.connected.emit(info, ser)
                else:
                    print('Connection error!')
        else:
            if self.__ser is not None:
                info = self.__ser_info
                self.__ser.close()
                ser = self.__ser
                self.__ser = None
                self.__ser_info = None
                self.disconnected.emit(info)
            self.__connect_button.setText('Connect')   
            self.__connect_button.setStyleSheet('QPushButton { background-color: green }')
            self.__port_list.setEnabled(True)
            self.__baud_list.setEnabled(True)
    
        

if __name__ == '__main__':
    # Note: This is really only for testing, but it's almost magical.
    #   Even just getting the line number that was causing a SEGFAULT made
    #   fixing the problem fast and easy.  This is worth remebering!
    import faulthandler
    faulthandler.enable()
    
    import sys
    
    class TestWidget(QtWidgets.QWidget):
        def __init__(self, parent=None):
            QtWidgets.QWidget.__init__(self, parent=parent)
            layout = QtWidgets.QVBoxLayout()
            self.ctrl = SerialControl(self)
            self.ctrl.connected.connect(self.process_connected)
            self.ctrl.disconnected.connect(self.process_disconnected)
            self.ctrl.ports_changed.connect(self.process_ports_changed)
            layout.addWidget(self.ctrl)
            self.search = QtWidgets.QPushButton('Num Ports Available?')
            self.search.clicked.connect(self.do_search)
            layout.addWidget(self.search)
            self.setLayout(layout)
            self.__ser = None
            
        @QtCore.pyqtSlot()
        def do_search(self):
            num_ports = self.ctrl.available_ports
            print('Number of ports available = {}'.format(num_ports))
            
        @QtCore.pyqtSlot(QtSerialPort.QSerialPortInfo, QtSerialPort.QSerialPort)
        def process_connected(self, info, ser):
            print('Connected to {}'.format(self.__fmt_serial_info(info)))
            self.__ser = ser
            self.__ser.readyRead.connect(self.__dump_ser)
    
        @QtCore.pyqtSlot(QtSerialPort.QSerialPortInfo)
        def process_disconnected(self, ser):
            print('Disconnected from {}'.format(self.__fmt_serial_info(ser)))
            self.__ser = None
            
        @QtCore.pyqtSlot(set, set)
        def process_ports_changed(self, added, removed):
            print('Added: {}, Removed: {}'.format(added, removed))   
            
        @QtCore.pyqtSlot()
        def __dump_ser(self):
            print('In dump ser')
            if self.__ser is not None:
                data = self.__ser.read(1000)
                print('*** {}'.format(data))
            
        def __fmt_serial_info(self, ser):
            return '{}: Manufacturer - {}, Description - {}'.format(ser.systemLocation(), ser.manufacturer(), ser.description())
        

    def main():
        app = QtWidgets.QApplication(sys.argv)
        win = TestWidget()
        win.show()
        app.exec_()

    main()
