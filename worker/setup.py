from setuptools import setup
import os


def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()


def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


setup(
    name="fleece-worker",
    version=get_version('fleece-worker/__init__.py'),
    description="fleece-worker",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="stneng",
    author_email="git@stneng.com",
    url="https://github.com/CoLearn-Dev/fleece-worker",
    packages=["fleece-worker"],
    install_requires=[
        "numpy",
        "torch",
        "fire",
        "sentencepiece",
        "fastapi",
        "uvicorn",
        "requests",
        "cryptography",
        "fleece-network==0.2.2"
    ],
    python_requires=">=3.10",
)
