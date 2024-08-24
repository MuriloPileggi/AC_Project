from setuptools import find_packages, setup

package_name = 'ros_python_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='muzaodamassa',
    maintainer_email='MuzaoDaMassaGames@gmail.com',
    description='A package for Python ROS 2 subscriber and Arduino communication.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'listener = ros_python_pkg.listener:main'
        ],
    },
)
