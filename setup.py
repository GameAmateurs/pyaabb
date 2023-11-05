from setuptools import setup, find_packages


setup(
    name="pyaabb",
    author="GameAmateurs",
    version="0.0.1.dev0",
    description="AABB collision in python",
    license="MIT",
    install_requires=[
        'numpy'
    ],
    packages=find_packages(),
    python_requires=">=3.8"
)