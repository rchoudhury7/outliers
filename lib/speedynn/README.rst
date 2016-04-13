========
speedynn
========

The speedynn package is a Python library that aims at accelerating feature selection for nearest neighbor models via modern many-core devices such as graphics processing units (GPUs). The implementation is based on `OpenCL <https://www.khronos.org/opencl/OpenCL>`_. 

=============
Documentation
=============

See the `documentation <http://speedynn.readthedocs.org>`_ for details and examples.

============
Dependencies
============

The speedynn package is tested under Python 2.6 and Python 2.7. The required Python dependencies are:

- NumPy >= 1.6.1

Further, `Swig <http://www.swig.org>`_, `OpenCL <https://www.khronos.org/opencl/OpenCL>`_, `setuptools <https://pypi.python.org/pypi/setuptools>`_, and a working C/C++ compiler need to be available. See the `documentation <http://speedynn.readthedocs.org>`_ for more details.

==========
Quickstart
==========

The package can easily be installed via pip via::

  pip install speedynn

To install the package from the sources, first get the current stable release via::

  git clone https://github.com/gieseke/speedynn.git

Afterwards, on Linux systems, you can install the package locally for the current user via::

  python setup.py install --user

On Debian/Ubuntu systems, the package can be installed globally for all users via::

  python setup.py build
  sudo python setup.py install

==========
Disclaimer
==========

The source code is published under the GNU General Public License (GPLv2). The authors are not responsible for any implications that stem from the use of this software.

