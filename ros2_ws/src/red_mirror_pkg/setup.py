from setuptools import find_packages, setup

package_name = 'red_mirror_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yuchen',
    maintainer_email='wangyuchen0217@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        	'red_mirror_dynamixel_node = red_mirror_pkg.red_mirror_dynamixel_node:main',
        	'red_mirror_esp32_node = red_mirror_pkg.red_mirror_esp32_node:main',
        ],
    },
)
