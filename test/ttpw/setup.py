#!/bin/env python
# -*- coding: utf-8 -*-
# Author: Laurent Pointal <laurent.pointal@limsi.fr> <laurent.pointal@laposte.net>

from distutils.core import setup
import sys
import io

setup(
    name='treetaggerwrapper',
    version='2.2.2',
    author='Laurent Pointal',
    author_email='laurent.pointal@limsi.fr',
    url='http://perso.limsi.fr/pointal/dev:treetaggerwrapper',
    download_url='https://sourcesup.renater.fr/projects/ttpw/',
    description='Wrapper for the TreeTagger text annotation tool from H.Schmid.',
    py_modules=['treetaggerwrapper', 'treetaggerpoll'],
    keywords=['tagger','treetagger','wrapper','text','annotation','linguistic'],
    license='GNU General Public License v3 or greater',
    requires=['six'],
    classifiers=[
                'Development Status :: 5 - Production/Stable',
                'Intended Audience :: Science/Research',
                'Natural Language :: English',
                'Operating System :: OS Independent',
                'Programming Language :: Python :: 2',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3',
                'Programming Language :: Python :: 3.4',
                'License :: OSI Approved :: GNU General Public License (GPL)',
                'Topic :: Scientific/Engineering',
                'Topic :: Scientific/Engineering :: Information Analysis',
                'Topic :: Software Development :: Libraries :: Python Modules',
                'Topic :: Text Processing :: Linguistic',
             ],
    long_description=io.open("README.txt", encoding='utf-8').read(),
    )

