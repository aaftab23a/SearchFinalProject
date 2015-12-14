#!/bin/env python
# -*- coding: utf-8 -*-
"""TreeTagger Python wrapper test module for multithreaded usage.

"""

from __future__ import print_function
from __future__ import unicode_literals

import unittest
# For my tests, not available on Python2
import concurrent.futures as cf
import threading
# Setup parent directory in sys.path.
import sys
from os import path

thedir = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, thedir)

# Now, import the module to test
import treetaggerwrapper


class MultithreadedTagging(unittest.TestCase):
    """Process same text from multiple threads, multiple times.
    """
    def setUp(self):
        self.tt = treetaggerwrapper.TreeTagger(TAGLANG='en')
        # The string to process.
        self.instr = "This is John's car."
        # The tokens we will send.
        self.tokstr = self.tt.tag_text(self.instr, prepronly=True)
        # The result we should have.
        self.tagres = self.tt.tag_text(self.tokstr, tagonly=True)

    def do_preproc(self):
        return self.tt.tag_text(self.instr, prepronly=True)

    def do_tagging(self):
        return self.tt.tag_text(self.tokstr, tagonly=True)

    def do_preproc_tagging(self):
        return self.tt.tag_text(self.instr)

    def check_one_fct(self, fct, output):
        executor = cf.ThreadPoolExecutor(max_workers=100)
        procs = []
        for x in range(1000):
            procs.append(executor.submit(fct))
        for p in procs:
            with self.subTest():
                res = p.result()
                self.assertEqual(res, output)

    def test_concurrent_preproc(self):
        """Concurrent preprocessing."""
        # Test one synchronous call.
        self.do_preproc()
        self.check_one_fct(self.do_preproc, self.tokstr)

    def test_concurrent_tagging(self):
        """Concurrent tagging."""
        # Test one synchronous call.
        self.do_tagging()
        self.check_one_fct(self.do_tagging, self.tagres)

    def test_concurrent_preproc_tagging(self):
        """Concurrent preprocessing+tagging."""
        # Test one synchronous call.
        self.do_preproc_tagging()
        self.check_one_fct(self.do_preproc_tagging, self.tagres)


if __name__ == '__main__':
    unittest.main()
