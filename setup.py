from setuptools import setup, find_packages

setup(
    name="autoif",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tqdm",
        # 其他依赖...
    ],
    entry_points={
        'console_scripts': [
            'autoif=autoif.cli.cli:main',
        ],
    },
)