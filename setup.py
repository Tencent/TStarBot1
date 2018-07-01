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
        'gym==0.9.4',
    ],
)
