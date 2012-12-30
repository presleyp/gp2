import sys

num = sys.argv[1]

if num < 2:
   with open('test1.txt', 'w') as f:
       f.write('test 1 worked')
elif num < 3:
   with open('test2.txt', 'w') as f:
       f.write('test 2 worked')
else:
   with open('test3.txt', 'w') as f:
       f.write('test 3 worked')
