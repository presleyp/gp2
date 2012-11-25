def con_general(infile):
    with open(infile, 'r') as f:
        text = f.readlines()
        constraints = [ccondense(cstrip(line)) for line in text if ...]
        conlengths = [len(constraint) for constraint in constraints]
        return conlengths

def cstrip(constraint):
    ...

def ccondense(constraint): # or just len?
    ...

import numpy
conlengths = con_general('...')
print numpy.mean(conlengths)
print numpy.sd(conlengths)
#print distribution or at least hist
