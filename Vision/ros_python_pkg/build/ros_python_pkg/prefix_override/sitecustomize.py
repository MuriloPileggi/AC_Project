import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/muzaodamassa/AC_Project/Vision/ros_python_pkg/install/ros_python_pkg'
