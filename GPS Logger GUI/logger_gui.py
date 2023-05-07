# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 15:47:33 2022

@author: marti
"""

# Note: This is really only for testing, but it's almost magical.
#   Even just getting the line number that was causing a SEGFAULT made
#   fixing the problem fast and easy.  This is worth remebering!
import faulthandler
faulthandler.enable()

import datetime
import sys

import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import PyQt5.QtWidgets as QtWidgets
import PyQt5.QtSerialPort as QtSerialPort

import log_parser
import serial_util

class ApplicationState:
    def __init__(self):
        self.ser = None
        self.connected = False
        self.download_samples = []
        

class DownloadWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(bool)
    progress = QtCore.pyqtSignal(int)
    parsed = QtCore.pyqtSignal(list)
    transmit = QtCore.pyqtSignal(str)
    timeout = QtCore.pyqtSignal()
    
    class EventHandler:

        def __init__(self, state):
            self.__state = state            
            self.running = True
            
        def reset(self):
            self.running = True
            
        def ack(self):
            self.__cmd('A')
            
        def nack(self):
            self.__cmd('N')
            
        def finished(self):
            self.running = False
            
        def __cmd(self, cmd):
            self.__state.ser.write(cmd.encode('ascii'))
            self.__state.ser.waitForBytesWritten(100)
            return
       
        
    def __init__(self, state, timeout_ms=2500):
        QtCore.QObject.__init__(self)
        self.__state = state
        self.__buffer = ''
        self.__log_parser_event_handler = DownloadWorker.EventHandler(state)
        self.__log_parser = log_parser.LogParser(self.__log_parser_event_handler)
        self.__running = True
        self.__timeout_timer = QtCore.QTimer(self)
        self.__timeout_ms = timeout_ms
        # self.__timeout_timer.timeout.connect(self.__timeout_handler)

    def reset(self):
        self.__buffer = ''
        self.__running = True
        self.__timeout_timer.stop()
        return
    
    def run(self):  
        print('self.__state.ser = {}'.format(self.__state.ser))
        self.__timeout_timer.start(self.__timeout_ms)
        success = True
        self.__cmd('D')
        
        while self.__log_parser_event_handler.running and success:
            # Read Data
            self.__state.ser.waitForReadyRead(100)
            new_data = self.__state.ser.readAll()
            new_data = bytes(new_data).decode('ascii')
            print(new_data)
            self.__buffer += new_data
#            print(new_data)            
            # print('Run Loop')
            # new_samples = self.__log_parser(self.__buffer)
            # self.__buffer = ''
            # if len(new_samples) > 0:
            #     # print('new samples = {}'.format(new_samples))
            #     self.parsed.emit(new_samples)                
                # self.__timeout_timer.start(self.__timeout_ms)
            success = (self.__timeout_timer.remainingTime() > 0)  
        self.finished.emit(success)
        return success
    
    def __cmd(self, cmd):
        self.__state.ser.write(cmd.encode('ascii'))
        self.__state.ser.waitForBytesWritten(100)
        return
        


class LogDownloadWidget(QtWidgets.QWidget):
    def __init__(self, state, parent=None):
        QtWidgets.QWidget.__init__(self, parent=parent)
        layout = QtWidgets.QGridLayout()
        self.__name_label = QtWidgets.QLabel('Model Name')
        self.__name_edit = QtWidgets.QLineEdit()
        layout.addWidget(self.__name_label, 0, 0, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(self.__name_edit, 0, 1, 1, 3)
        self.__engine_label = QtWidgets.QLabel('Engines')
        self.__engine_edit = QtWidgets.QLineEdit()
        layout.addWidget(self.__engine_label, 1, 0, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(self.__engine_edit, 1, 1, 1, 3)        
        self.__mass_label = QtWidgets.QLabel('Liftoff Mass')
        self.__mass_edit = QtWidgets.QLineEdit()
        layout.addWidget(self.__mass_label, 2, 0, alignment=QtCore.Qt.AlignRight)
        layout.addWidget(self.__mass_edit, 2, 1, 1, 3)            
        self.__download_button = QtWidgets.QPushButton('Download')
        self.__save_button = QtWidgets.QPushButton('Save')
        self.__erase_button = QtWidgets.QPushButton('Erase')
        layout.addWidget(self.__download_button, 3, 0)
        layout.addWidget(self.__save_button, 3, 1)
        layout.addWidget(self.__erase_button, 3, 2)
        self.__display_window = QtWidgets.QTextEdit()
        layout.addWidget(self.__display_window, 4, 0, 4, 4)
        self.__download_button.clicked.connect(self.__begin_download)
        self.__save_button.clicked.connect(self.__save_download)
        self.__erase_button.clicked.connect(self.__begin_erase)
        self.setLayout(layout)
        self.__download_thread = QtCore.QThread()
        self.__download_worker = DownloadWorker(state)
        self.__download_thread.started.connect(self.__download_worker.run)
        self.__download_worker.parsed.connect(self.__update_samples)
        self.__download_worker.finished.connect(self.__end_download)

        self.__state = state
        
        
    def __begin_download(self):
        print('Begin Download')
        self.__download_thread.start()
    
    def __update_samples(self, samples):
        print('New Samples - {}'.format(samples))
        self.__state.download_samples += samples
        return
    
    def __end_download(self, success):
        print('Ended Download With Success = {}'.format(success))
        self.__download_thread.quit()
        return
    
    def __save_download(self):
        print('Run Save Download')
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        filename, ext = QtWidgets.QFileDialog.getSaveFileName(self, 'Download File...', '', 'Log Data (*.log);;All Files (*);;Comma-Separated (.csv)', options=options)
        # TODO: USE THE EXTENSION RETURN TO AUTOMATICALLY ADD THE EXTENSION IF IT IS NOT ON THE FILENAME ALREADY
        
        if filename:
            print('Filename = {}'.format(filename))

    def __begin_erase(self):
        self.__state.ser.write(b'E')
        

class ConnectionWidget(QtWidgets.QWidget):
    def __init__(self, state, parent=None):
        QtWidgets.QWidget.__init__(self, parent=parent)
        layout = QtWidgets.QVBoxLayout()
        self.__ctrl = serial_util.SerialControl(self)
        self.__ctrl.connected.connect(self.process_connected)
        self.__ctrl.disconnected.connect(self.process_disconnected)
#        self.ctrl.ports_changed.connect(self.process_ports_changed)
        layout.addWidget(self.__ctrl)
#        self.search = QtWidgets.QPushButton('Num Ports Available?')
#        self.search.clicked.connect(self.do_search)
#        layout.addWidget(self.search)
        self.setLayout(layout)
        self.__state = state

    @property
    def connection_control(self):
        return self.__ctrl
    
    @QtCore.pyqtSlot(QtSerialPort.QSerialPortInfo, QtSerialPort.QSerialPort)
    def process_connected(self, info, ser):
#        print('Connected to {}'.format(self.__fmt_serial_info(info)))
        self.__state.ser = ser
        # self.__state.ser.readyRead.connect(self.__dump_ser)

    @QtCore.pyqtSlot(QtSerialPort.QSerialPortInfo)
    def process_disconnected(self, ser):
#        print('Disconnected from {}'.format(self.__fmt_serial_info(ser)))
        self.__state.ser = None
            
    # @QtCore.pyqtSlot()
    # def __dump_ser(self):
    #     if self.__state.ser is not None:
    #         data = self.__state.ser.read(1000) # TODO: DECIDE IF THIS READ DEPTH NEEDS TO BE PROGRAMMABLE, OR CHANGED
    #         print('*** {}'.format(data))        
    
    
class ApplicationWidget(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        
        self.__state = ApplicationState() 
        self.__tabs = QtWidgets.QTabWidget(self)
        self.__connection_tab = ConnectionWidget(self.__state, self)
        self.__download_tab = LogDownloadWidget(self.__state, self)
#        self.tabs.resize(300,200)
        
        # Add tabs
        self.__tabs.addTab(self.__connection_tab,"Serial Connection")
        self.__tabs.addTab(self.__download_tab,"Log Download")
        self.__tabs.setTabEnabled(1, False)
        
        self.setCentralWidget(self.__tabs)
        
        self.__status_bar = QtWidgets.QStatusBar()
        self.__status_label = QtWidgets.QLabel('Status')
        self.__status_bar.addWidget(self.__status_label)
        self.setStatusBar(self.__status_bar)
        
        self.__connection_tab.connection_control.connected.connect(self.__process_connect)
        self.__connection_tab.connection_control.disconnected.connect(self.__process_disconnect)
        
    def __process_connect(self):
        self.__tabs.setTabEnabled(1, True)
        
    def __process_disconnect(self):
        self.__tabs.setTabEnabled(1, False)

        

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ApplicationWidget()
    win.show()
    app.exec_()
    app.exit()

    
if __name__ == '__main__':
    main()