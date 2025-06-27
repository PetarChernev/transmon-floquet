from setuptools import setup, find_packages

setup(
    name="transmon_floquet",           # replace with your project’s import name
    version="0.1.0",
    py_modules=[
        "transmon",
        "optimization"
    ],
    packages=find_packages(),   # auto-discovers your top-level package(s)
    python_requires=">=3.13",    # adjust to your minimum supported Python
)