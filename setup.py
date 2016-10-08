import os
import sys
import io
from setuptools import setup, find_packages
from setuptools.command.install import install as _install

from download_nltk_data import download_data

if sys.version_info[:2] < (2, 7):
    raise Exception('This version of gensim needs Python 2.7 or later.')


def readfile(fname):
    path = os.path.join(os.path.dirname(__file__), fname)
    return io.open(path, encoding='utf8').read()




class Install(_install):
    def run(self):
        _install.do_egg_install(self)
        download_data()


setup(name='text-analysis',
      version='0.1',
      description='UMN Research on health journals.',
      long_description=readfile('README.md'),
      url='https://github.com/robert-giaquinto/text-analysis',
      author='UMN',
      license='MIT',
      packages=find_packages(),
      test_suite='src.tests',
      include_package_data=True,
      cmdclass={'install': Install},
      install_requires=['nltk'],
      setup_requires=['nltk'],
      zip_safe=False)
