'''
Created on 01.08.2016

@author: Fabian Gieseke
'''

def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration('speedynn', parent_package, top_path)
    config.add_subpackage('classification', subpackage_path='classification')
    config.add_subpackage('tests')
    config.add_subpackage('util')    

    return config

if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
