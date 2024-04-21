#include "serialCommunication.hpp"

void serialCommunication::writeToSerialPort(const std::string& dataToSend, const bool& send) 
{
    if (send) {
        // Open the serial port
        int serialPort = open(serialPortFileName.c_str(), O_RDWR);

        // Check if serial port was correctly opened, if not exit the method
        if (serialPort == -1) {
            std::cerr << "Error: Failed to open serial port" << std::endl;
            return;
        }

        // Set serial port settings (baud rate)
        struct termios tty;
        tcgetattr(serialPort, &tty);
        cfsetospeed(&tty, B9600); // Set baud rate to 9600
        cfsetispeed(&tty, B9600);
        tcsetattr(serialPort, TCSANOW, &tty);

        // Send data to arduino
        write(serialPort, dataToSend.c_str(), dataToSend.length());

        // Close serial port
        close(serialPort);

        // Give some time for arduino to process data
        sleep(2);

        return;
    }
}

void serialCommunication::readFromSerialPort()
{
    
}