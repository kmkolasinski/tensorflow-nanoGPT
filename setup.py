from setuptools import find_packages, setup
from wise_geometry.version import version

setup(
    name="tf_nano_gpt",
    version=version,
    description="",
    url="https://github.com/kmkolasinski/tensorflow-nanoGPT",
    author="Krzysztof Kolasinski",
    author_email="kmkolasinski@gmail.com",
    packages=find_packages(exclude=["tests", "notebooks"]),
    include_package_data=False,
    zip_safe=False,
    install_requires=[],
)
