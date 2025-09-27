from setuptools import setup

package_name = 'py_reservoir_rossler_eval'

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
    description='A simple Reservoir Computing model with ROS 2 Python',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'agent = py_reservoir_rossler_eval.agent:main',
            'edge = py_reservoir_rossler_eval.edge:main',
        ],
    },
)

