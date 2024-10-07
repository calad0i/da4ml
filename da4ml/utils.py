from collections import namedtuple

DAState = namedtuple('DAState', ['csd', 'variables', 'op_codes', 'pairs', 'score', 'kernel'])
Score = namedtuple('Score', ['potential', 'realized', 'lost', 'value'])
OpCode = namedtuple('OPCode', ['pos0', 'pos1', 'shift0', 'shift1', 'sign0', 'sign1'])
