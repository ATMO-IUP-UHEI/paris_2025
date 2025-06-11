from setuptools import setup, find_packages

setup(
    name="paris_2025",
    version="0.1.0",
    description="A project for CO2 modelling in paris.",
    author="Robert Maiwald",
    author_email="robert.maiwald@uni-heidelberg.de",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    python_requires=">=3.7",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
