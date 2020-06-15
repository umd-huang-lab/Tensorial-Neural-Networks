
#import sys
#import os
## \todo need to figure out where to put the lib/tnn.so binary so that it adheres to
##       usual import semantics
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'bin'))
#import tnnlib

from .parse_conv_einsum import * 
#from .conv_einsumfunc import *
from .torch_conv_einsumfunc import *
