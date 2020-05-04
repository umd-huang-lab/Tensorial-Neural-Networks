
import sys
import os
# \todo need to figure out where to put the lib/tnn.so binary so that it adheres to
#       usual import semantics
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lib'))
import tnnlib

# just for testing
from .conv_einsumfunc import *
