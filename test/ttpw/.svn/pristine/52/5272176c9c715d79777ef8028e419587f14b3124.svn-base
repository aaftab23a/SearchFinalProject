from __future__ import print_function
from __future__ import unicode_literals

# Import the development version of treetaggerwrapper.
import sys
sys.path.insert(0, "..")

import sys
import time

JOBSCOUNT = 10000

def start_test(n=None):
    start = time.time()
    import treetaggerpoll

    # Note: print() have been commented, you may uncomment them to see progress.
    p = treetaggerpoll.TaggerProcessPoll(workerscount=n, TAGLANG="en")
    res = []

    text = "This is Mr John's own house, it's very nice. " * 40

    print("Creating jobs")
    for i in range(JOBSCOUNT):
        # print("\tJob", i)
        res.append(p.tag_text_async(text))

    print("Waiting for jobs to complete")
    for i, r in enumerate(res):
        # print("\tJob", i)
        r.wait_finished()
        # print(str(r.result)[:50])
        res[i] = None   # Loose Job reference - free it.

    p.stop_poll()
    print("Finished after {:0.2f} seconds elapsed".format(time.time() - start))

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        nproc = int(sys.argv[1])
    else:
        nproc = None
    start_test(nproc)

