import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="example-pkg-YOUR-USERNAME-HERE", # Replace with your own username
    version="0.0.1",
    author="Ashton Antoun",
    author_email="Ashton.Antoun@gmail.com",
    description="Algorithm to find the optimum volatility for the Black & Scholes formula for analysing stock options",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AshtonGit/Black&ScholesVolatilityOptimisation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)