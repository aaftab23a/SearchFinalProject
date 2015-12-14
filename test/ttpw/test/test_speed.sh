#!/bin/sh

# Retrieve corpus data if not present.
python3 prepare.py

# Run once to populate OS cache.
python process_once.py

# Timed process
echo "Using Python 2 with standard chunking:"
time -p python process_once.py $1

echo "Using Python 3 with standard chunking:"
time -p python3 process_once.py $1

# Timed process
echo "Using Python 2 with external chunking:"
time -p python process_once.py external $1

echo "Using Python 3 with external chunking:"
time -p python3 process_once.py external $1
