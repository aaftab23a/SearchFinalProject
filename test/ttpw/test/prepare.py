#!/usr/bin/python3
# -*- encoding: utf-8 -*-

import os
from os import path as osp
import urllib.request

CORPUS = [
    ("https://www.gutenberg.org/files/29772/29772-0.txt", "notevoltcond.txt"),
    ("https://www.gutenberg.org/ebooks/27828.txt.utf-8", "souvanecileelbe.txt"),
    ("https://www.gutenberg.org/files/32297/32297-0.txt", "philozoolavantdarwin.txt"),
    ]

# Build data dir if necessary.
if not osp.isdir("data"):
    os.mkdir("data")

# Download large text files.
for url, filename in CORPUS:
    fname = osp.join('data', filename)
    if not osp.isfile(fname):
        print("Download {}, save as {}".format(url, fname))
        response = urllib.request.urlopen(url)
        data = response.read()
        # Note: selected files are utf8, we write them as is.
        with open(fname, "wb") as f:
            f.write(data)

# List these filenames for process_once.
with open("fileslist.txt", "w", encoding="utf-8") as f:
    for url, filename in CORPUS:
        f.write(osp.join('data', filename) + "\n")

