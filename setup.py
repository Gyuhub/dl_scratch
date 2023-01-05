import io
from setuptools import find_packages, setup

# Read in the README for the long description on PyPI
def long_description():
    with io.open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme

setup(name='dl_scratch',
      version='0.1',
      description='python programming practices with deep learning from scratch',
      long_description=long_description(),
      url='https://github.com/Gyuhub/dl_scratch',
      license='MIT',
      author='Gyuhub',
      author_email='alsrb0820@gmail.com',
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python :: 3.9',
          ],
      install_requires=[
          'numpy',
          'matplotlib',
          ],
      zip_safe=False)
