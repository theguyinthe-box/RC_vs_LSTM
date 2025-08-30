from setuptools import setup

package_name = 'py_lstm_rossler'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Felix Grote',
    maintainer_email='felix.grote@unibw.de',
    description='An agent-edge model using an LSTM with ROS 2 (Python) for comparison with a reservoir computing baseline',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'agent = py_lstm_rossler.agent:main',
            'edge = py_lstm_rossler.edge:main',
        ],
    },
)

