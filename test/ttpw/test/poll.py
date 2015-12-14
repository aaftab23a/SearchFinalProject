from __future__ import print_function
from __future__ import unicode_literals

import sys
import io

# Import the development version of treetaggerwrapper.
sys.path.insert(0, "..")
import treetaggerwrapper as ttpw

#ttpw.enable_debugging_log()

p = ttpw.TaggerPoll()
JOBSCOUNT = 10000
res = []

text = "This is Mr John's own house, it's very nice."
#text = io.open("data/notevoltcond.txt", encoding='utf-8').read()

print("Creating jobs")
for i in range(JOBSCOUNT):
    print("\tJob", i)
    res.append(p.tag_text_async(text))

print("Waiting for jobs to be completed")
for i, r in enumerate(res):
    print("\tJob", i)
    r.wait_finished()
    print(str(r.result)[:50])
    res[i] = None   # Loose Job reference - free it.

p.stop_poll()
print("Finished")