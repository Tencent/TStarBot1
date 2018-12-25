from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup


description = """Macro-action based StarCraft-II learning environment."""

setup(
    name='sc2learner',
    version='0.1',
    description='Macro-action based StarCraft-II learning environment.',
    long_description=description,
    author='Tencent AI Lab',
    author_email='xinghaisun@tencent.com',
    keywords='sc2learner StarCraft AI',
    url='https://github.com/Tencent-Game-AI/sc2learner',
    packages=[
        'sc2learner',
        'sc2learner.agents',
        'sc2learner.envs',
        'sc2learner.utils',
        'sc2learner.bin',
    ],
    install_requires=[
        'gym==0.10.5',
        'torch==0.4.0',
        'tensorflow>=1.4.1',
        'joblib',
        'pyzmq'
    ]
)
