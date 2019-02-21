import sys
import os

path = os.path.dirname(os.path.realpath(__file__))

if path not in sys.path:
    sys.path.insert(0, path)

