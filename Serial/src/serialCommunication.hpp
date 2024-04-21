#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <termios.h> // Serial port settings 
#include <fcntl.h> // File control options
#include <unistd.h> // sleep()


class serialCommunication
{
private:
    const std::string serialPortFileName = "/dev/ttyACM0";
public:
    // Method to write to serial port
    void writeToSerialPort(const std::string& dataToSend, const bool& send);

    // Method to read from serial port  
    void readFromSerialPort();
};
