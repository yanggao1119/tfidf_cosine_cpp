import sys

"""
for each doc, convert from multi-line uci sparse bag-of-words format to one-line format; skipped doc is represented with blank line as placeholder

see following example, note that first three lines are placeholders for meta data, not important for this conversion

$ cat toy_docword2oneline.txt 
1231
131
131
1 11 111
1 22 222

2 33 333
3 45 555
5 88 888
9 12 34
9 56 777
10 1 1
$ cat toy_docword2oneline.txt | python get_docword2oneline.py 
11 111 22 222
33 333
45 555

88 888



12 34 56 777
1 1
=======end of output=========

"""


prev_line_ind = 1
prev_line_list = []

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    linesplit = line.split()
    if len(linesplit) == 3:
        line_ind, w_ind, w_c = linesplit
        line_ind = int(line_ind)

        # case 1: prev line "1 x x", curr line "1 y y"
        if line_ind == prev_line_ind:
            prev_line_list.append((w_ind, w_c))
        # case 2: prev line "1 x x", curr line "2 y y"
        elif line_ind == prev_line_ind + 1:
            print ' '.join(t[0] + ' ' + t[1] for t in prev_line_list)
            prev_line_ind = line_ind
            prev_line_list = [(w_ind, w_c)]
        # case 3: prev line "1 x x" curr line "5 y y"
        elif line_ind > prev_line_ind +1:
            print ' '.join(t[0] + ' ' + t[1] for t in prev_line_list)
            print '\n'*(line_ind - prev_line_ind - 2)
            prev_line_ind = line_ind
            prev_line_list = [(w_ind, w_c)]
print ' '.join(t[0] + ' ' + t[1] for t in prev_line_list)
  
