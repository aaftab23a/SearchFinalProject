#!/bin/env python
# -*- coding: utf-8 -*-
"""TreeTagger Python wrapper test module.

Note: You must run tests under Python3, as subTest is not
(as write time) in Python2 test package.
"""

from __future__ import print_function
from __future__ import unicode_literals

import unittest
# from test import test_support

# Setup parent directory in sys.path.
import sys
from os import path

thedir = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, thedir)

# Now, import the module to test
import treetaggerwrapper


class TTStartTestCase(unittest.TestCase):
    """Try to start TreeTagger - if this fail..."""

    def test_start_tagger(self):
        tts = treetaggerwrapper.TreeTagger()


class TextPreprocessing(unittest.TestCase):
    """Preprocess some texts - must be subclassed with providing."""
    taggers = {}

    def __init__(self, lang, *args):
        unittest.TestCase.__init__(self, *args)
        self._sample_and_res = []
        # sample_and_res is a list of tuples, each tuple has two items, the
        # first one is the string to preprocess, the second one is the attended
        # result (a list of strings). Alls trings must be unicode.
        self.__lang = lang

    def setUp(self):
        if self.__lang not in self.taggers:
            self.taggers[self.__lang] = treetaggerwrapper.TreeTagger(TAGLANG=self.__lang)
        self.tt = self.taggers[self.__lang]

    def runTest(self):
        for i, (sample, res, _) in enumerate(self._sample_and_res):
            with self.subTest(i=i):
                ttres = self.tt._prepare_text(sample)
                self.assertEqual(ttres, res)

class TextProcessing(unittest.TestCase):
    """Preprocess some texts - must be subclassed with providing."""
    taggers = {}

    def __init__(self, lang, *args):
        unittest.TestCase.__init__(self, *args)
        self._sample_and_res = []
        # sample_and_res is a list of tuples, each tuple has two items, the
        # first one is the string to preprocess, the second one is the attended
        # result (a list of strings). Alls trings must be unicode.
        self.__lang = lang

    def setUp(self):
        if self.__lang not in self.taggers:
            self.taggers[self.__lang] = treetaggerwrapper.TreeTagger(TAGLANG=self.__lang)
        self.tt = self.taggers[self.__lang]

    def runTest(self):
        for i, (sample, _, res) in enumerate(self._sample_and_res):
            with self.subTest(i=i):
                ttres = self.tt.tag_text(sample)
                self.assertEqual(ttres, res)


class EnglishPreprocessing(TextPreprocessing):
    """Test english preprocessing (and chunking).
    """

    def __init__(self, *args):
        TextPreprocessing.__init__(self, "en", *args)
        self._sample_and_res = ENGLISH_TESTS

class EnglishProcessing(TextPreprocessing):
    """Test english preprocessing (and chunking).
    """

    def __init__(self, *args):
        TextPreprocessing.__init__(self, "en", *args)
        self._sample_and_res = ENGLISH_TESTS


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TTStartTestCase('test_start_tagger'))
    suite.addTest(EnglishPreprocessing())
    suite.addTest(EnglishProcessing())
    return suite


def test_main():
    unittest.TextTestRunner(verbosity=2).run(suite())

    # test_support.run_unittest(
    # TTStartTestCase,
    # EnglishPreprocessingTestCase
    # )


# Each set of test strings must include three parts:
# 1) input sentence
# 2) attended chunking
# 3) attended tagging
ENGLISH_TESTS = [
            # Basic sentence.
            ("Hello, Mr Young.",
             ["Hello", ",", "Mr", "Young", "."],    # Preprocessing chunks
                ["Hello\tUH\tHello",
                ",\t,\t,",
                "Mr\tNP\tMr",
                "Young\tNP\tYoung",
                ".\tSENT\t."]
            ),
            # DNS in a sentence.
            ("See domain www.truc.com for informations.",
             ["See", "domain", "replaced-dns", '<repdns text="www.truc.com" />',
              "for", "informations", "."],
                ['See\tVV\tsee',
                 'domain\tNN\tdomain',
                 'replaced-dns\tNNS\treplaced-dns',
                 '<repdns text="www.truc.com" />',
                 '.\tSENT\t.',
                 'Big\tJJ\tbig',
                 'site\tNN\tsite',
                 '.\tSENT\t.']
            ),
            # DNS ending a sentence.
            ("See domain www.truc.com. Big site.",
             ["See", "domain", "replaced-dns", '<repdns text="www.truc.com" />',
              ".", "Big", "site", "."],
                ['A\tDT\ta',
                 'short\tJJ\tshort',
                 'replaced-dns\tNNS\treplaced-dns',
                 '<repdns text="domain.name" />',
                 '.\tSENT\t.']
            ),
            # Short DNS.
            ("A short domain.name.",
             ["A", "short", "replaced-dns", '<repdns text="domain.name" />', "."],
                []
            ),
            # Email in a sentence.
            ("My email is laurent.pointal@limsi.fr at work.",
             ["My", "email", "is", "replaced-email",
              '<repemail text="laurent.pointal@limsi.fr" />',
              "at", "work", "."],
                ['My\tPP$\tmy',
                 'email\tNN\temail',
                 'is\tVBZ\tbe',
                 'replaced-email\tNN\treplaced-email',
                 '<repemail text="laurent.pointal@limsi.fr" />',
                 'at\tIN\tat',
                 'work\tNN\twork',
                 '.\tSENT\t.']
            ),
            # Email with a post-dot ending the sentence.
            ("My email is laurent.pointal@limsi.fr. Yes it is.",
             ["My", "email", "is", "replaced-email",
              '<repemail text="laurent.pointal@limsi.fr" />',
              ".", "Yes", "it", "is", "."],
                ['My\tPP$\tmy',
                 'email\tNN\temail',
                 'is\tVBZ\tbe',
                 'replaced-email\tNN\treplaced-email',
                 '<repemail text="laurent.pointal@limsi.fr" />',
                 '.\tSENT\t.',
                 'Yes\tUH\tyes',
                 'it\tPP\tit',
                 'is\tVBZ\tbe',
                 '.\tSENT\t.']
            ),
            # Acronym, including a diacritic char, and 'nt before a final dot.
            ( "I don't work for B.À.D.C.O.M.P. No, I don't.",
                ["I", "do", "n't", "work", "for", "B.À.D.C.O.M.P.",
                 "No", ",", "I", "do", "n't", "."],
                    ['I\tPP\tI',
                     'do\tVVP\tdo',
                     "n't\tRB\tn't",
                     'work\tVV\twork',
                     'for\tIN\tfor',
                     'B.À.D.C.O.M.P.\tNP\tB.À.D.C.O.M.P.',
                     'No\tNP\tNo',
                     ',\t,\t,',
                     'I\tPP\tI',
                     'do\tVVP\tdo',
                     "n't\tRB\tn't",
                     '.\tSENT\t.']
            ),
            # English possessive case.
            ( "Mister Jone's help is needed.",
                [ "Mister", "Jone", "'s", "help", "is", "needed", "."],
                    ['Mister\tNP\tMister',
                     'Jone\tNP\tJone',
                     "'s\tPOS\t's",
                     'help\tNN\thelp',
                     'is\tVBZ\tbe',
                     'needed\tVVN\tneed',
                     '.\tSENT\t.']
            ),
            # English possessive case at end.
            ( "Its Marcel's.",
             [ "Its", "Marcel", "'s", "."],
                ['Its\tPP$\tits', 'Marcel\tNP\tMarcel', "'s\tPOS\t's", '.\tSENT\t.']
             ),
             # Special forms from abbreviations (english file) which fail with
             # perl chunker. Containing signs at end and in middle.
            ( "You are pro-U.S.",
             [ "You", "are", "pro-U.S."],
                ['You\tPP\tyou', 'are\tVBP\tbe', 'pro-U.S.\tJJ\tpro-U.S.']
            ),
            ( "The president-U.S. is with us.",
             ['The', 'president-U.S.', 'is', 'with', 'us', '.'],
                ['The\tDT\tthe',
                 'president-U.S.\tNN\tpresident-U.S.',
                 'is\tVBZ\tbe',
                 'with\tIN\twith',
                 'us\tPP\tus',
                 '.\tSENT\t.']
            ),
            ]

if __name__ == '__main__':
    # unittest.main()
    test_main()

