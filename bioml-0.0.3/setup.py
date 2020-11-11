import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bioml", # Replace with your own username
    version="0.0.3",
    author="Vincent Mendez",
    author_email="vincent.mendez@epfl.ch",
    description="Toolbox providing basic functions for bio-signal (pre)processing and machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vammendez/The-Awesome-App/tree/Development",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)