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

import serial_util

class ApplicationState:
    def __init__(self):
        self.ser = None
        self.connected = False
        
        
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
        self.__download_button = QtWidgets.QPushButton('Download')
        self.__save_button = QtWidgets.QPushButton('Save')
        self.__erase_button = QtWidgets.QPushButton('Erase')
        layout.addWidget(self.__download_button, 2, 0)
        layout.addWidget(self.__save_button, 2, 1)
        layout.addWidget(self.__erase_button, 2, 2)
        self.__display_window = QtWidgets.QTextEdit()
        layout.addWidget(self.__display_window, 3, 0, 4, 4)
        self.setLayout(layout)
        self.__state = state

    
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
        
    @QtCore.pyqtSlot(QtSerialPort.QSerialPortInfo, QtSerialPort.QSerialPort)
    def process_connected(self, info, ser):
#        print('Connected to {}'.format(self.__fmt_serial_info(info)))
        self.__state.ser = ser
        self.__state.ser.readyRead.connect(self.__dump_ser)

    @QtCore.pyqtSlot(QtSerialPort.QSerialPortInfo)
    def process_disconnected(self, ser):
#        print('Disconnected from {}'.format(self.__fmt_serial_info(ser)))
        self.__state.ser = None
            
    @QtCore.pyqtSlot()
    def __dump_ser(self):
        if self.__ser is not None:
            data = self.__ser.read(1000) # TODO: DECIDE IF THIS READ DEPTH NEEDS TO BE PROGRAMMABLE, OR CHANGED
            print('*** {}'.format(data))        
        

class ApplicationWidget(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        
        self.__state = None  # TODO: THIS IS WHERE THE STATE VARIABLE NEEDS TO GO 
        self.__tabs = QtWidgets.QTabWidget(self)
        self.__connection_tab = ConnectionWidget(self.__state, self)
        self.__download_tab = LogDownloadWidget(self.__state, self)
#        self.tabs.resize(300,200)
        
        # Add tabs
        self.__tabs.addTab(self.__connection_tab,"Serial Connection")
        self.__tabs.addTab(self.__download_tab,"Log Download")
        
        self.setCentralWidget(self.__tabs)
        
        self.__status_bar = QtWidgets.QStatusBar()
        self.__status_label = QtWidgets.QLabel('Status')
        self.__status_bar.addWidget(self.__status_label)
        self.setStatusBar(self.__status_bar)
        

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ApplicationWidget()
    win.show()
    app.exec_()
    app.exit()

    
if __name__ == '__main__':
    main()