from setuptools import find_packages, setup

package_name = 'init_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='muzaodamassa',
    maintainer_email='MuzaoDaMassaGames@gmail.com',
    description='A package for Python ROS 2 to initialize all other ROS 2 packages.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'init_node = init_pkg.init_node:main'
        ],
    },
)
