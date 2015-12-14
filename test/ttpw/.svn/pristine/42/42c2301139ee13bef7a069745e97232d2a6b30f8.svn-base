# -*- encoding: utf-8 -*-

from __future__ import print_function
from __future__ import unicode_literals

import pprint
import sys

sys.path.insert(0, '..')

EXTERNAL = ("external" in sys.argv)
PROFILING = False
TEST_BIGFILES = ("sentence" not in sys.argv)

# To compare with Alexandre Delanoë & Mathieu Rodic chunking regexp.
if EXTERNAL:
    import re
    sentence_exp = r'''(?x) # set flag to allow verbose regexps
        (?:[A-Z])(?:\.[A-Z])+\.? # abbreviations, e.g. U.S.A.
        | \w+(?:-\w+)* # words with optional internal hyphens
        | \$?\d+(?:\.\d+)?%? # currency and percentages, e.g. $12.40, 82%
        | \.\.\. # ellipsis
        | [][.,;"'?():-_`] # these are separate tokens
        '''
    sentence_re = re.compile(sentence_exp, re.UNICODE | re.MULTILINE | re.DOTALL)
    def chunkfct(tagger, textlist):
        res = []
        for text in textlist:
            res.extend(sentence_re.findall(text))
        return res
else:
    chunkfct = None

if PROFILING:
    import cProfile

import treetaggerwrapper as ttpw
#ttpw.enable_debugging_log()

tt = ttpw.TreeTagger(TAGLANG='fr', CHUNKERPROC=chunkfct)

if TEST_BIGFILES:
    import io

    with io.open("fileslist.txt", encoding="utf-8") as f:
        flist = f.read().splitlines()

    for fname in flist:
        # print("Tagging:", fname)
        if PROFILING:
            cProfile.run("res = tt.tag_file(fname)")
            break   # Just process one file.
        else:
            res = tt.tag_file(fname)
            # Uncomment to save result and diff data.
            #with io.open(fname+str(sys.version_info.major+['_internal', '_external'][EXTERNAL]),
            #               "w", encoding='utf-8') as f:
            #    f.writelines(res)
else:
    sentence = """
                Les sanglots longs
                Des violons
                De l'automne
                Blessent mon cœur
                D'une langueur
                Monotone.

                Tout suffocant
                Et blême, quand
                Sonne l'heure,
                Je me souviens
                Des jours anciens
                Et je pleure

                Et je m'en vais
                Au vent mauvais
                Qui m'emporte
                Deçà, delà,
                Pareil à la
                Feuille morte.
                """
    for i in range(10000):
        res = tt.tag_text(sentence)
        #pprint.pprint(res)
