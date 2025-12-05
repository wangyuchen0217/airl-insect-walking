from setuptools import setup

package_name = 'robot_policy'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','numpy'],
    zip_safe=True,
    maintainer='yuchen',
    maintainer_email='you@example.com',
    description='A minimal policy runner that outputs joint position commands.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_node = robot_policy.policy_node:main',
            'fake_joint_states = robot_policy.fake_joint_states:main',
            'fake_imu_states = robot_policy.fake_imu_states:main',
            'fake_contact_states = robot_policy.fake_contact_states:main',
        ],
    },
)
