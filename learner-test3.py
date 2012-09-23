import numpy
from CON import *
from featuredict import *
from GEN import *
from mapping import *
import sys, os

#localpath = '/'.join(sys.argv[0].split('/')[:-1])
#os.chdir(localpath)


fd = FeatureDict('TurkishFeaturesWithNA.csv')
wordvoi = Mapping(fd, ['1','+kurd+','kurt','stem change voi 1 d'])
wordvoi.to_data()
wordvoc = Mapping(fd, ['1','+tad+im','tadwm','change back -1 i'])
wordvoc.to_data()
print 'stems', wordvoi.stem, wordvoc.stem
print wordvoi.ur[0]
assert wordvoi.in_stem(0) == True
assert wordvoc.in_stem(2) == True
assert wordvoi.in_stem(4) == False
#print wordvoi
#for change in wordvoi.changes:
    #print 'change', change
    #print 'wordvoi.value', change.value
#print wordvoc
faithfulword = Mapping(fd, ['1', '+kurd+', 'kurd', 'none'])
faithfulword.to_data()

dgen = DeterministicGen(fd)
with open('TurkishInput4.csv', 'r') as f:
    lines = csv.reader(f)
    for line in list(lines)[1100:1101]:
        print line
        word = Mapping(fd, line)
        word.to_data()
        #print word
        #voicing = dgen.change_voicing(word)
        #print voicing
        ##if voicing:
            ##assert 'change -voi' in voicing.changes
        ##print dgen.make_new_mapping(word, 0, set([2,1]))
        #for mapping in dgen.change_vowels(word):
            #print mapping
        #print 'faithful:', dgen.make_faithful_cand(word)[0]
        #print 'negatives:'
        #for negative in dgen.ungrammaticalize(word):
            #print negative
        #print 'done'
        negatives = dgen.ungrammaticalize(word)
        word.add_boundaries()
        word.set_ngrams()
        print word
        print negatives[3]
        mark = MarkednessAligned(fd, .5, [word, negatives[2]])
        print mark
        faith = Faithfulness([word, negatives[3]], fd, True)
        print faith
assert [] == dgen.make_faithful_cand(faithfulword)

#inputs = Input(fd, 'TurkishInput4.csv', True, 'deterministic')
#tablengths = []
#for tableau in inputs.allinputs:
    #tablengths.append(len(tableau))
    ##for mapping in tableau:
        ##print mapping
#print numpy.mean(tablengths) #4.2
#print max(tablengths) # 6
#print min(tablengths) # 2

#TODO should i have cands with both changes?
#TODO test CON, EVAL
