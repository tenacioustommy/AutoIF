from setuptools import setup, find_packages

def read_requirements(filename='requirements.txt'):
    with open(filename) as f:
        return [line.strip() for line in f.read().splitlines()
                if not line.startswith('#') and line.strip()]

setup(
    name="autoif",
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'autoif=autoif.cli.cli:main',
        ],
    },
)