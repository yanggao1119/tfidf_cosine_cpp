#!/bin/bash

#NOTE: uncomment to each demo
# then run:      
# $sh run_examples.sh 

mkdir -p exp

# 1. train tfidf on toy data in uci format, then input query doc from STDIN and output cosine similarity result to STDOUT. For query you can type one-line compact docword format, such as "1 2 3 4" and hit enter. By default output top 50 (can be changed by switch)
#./tfidf_cosine -d data_uci/docword.lovecat.txt

# 2. as 1., yet using uci nips data and using files at both ends of the pipe, i.e., query_input >> STDIN and STDOUT >> query_out
## running sanity check to see if each doc predicts itself as the most similar, with cosine similarity as 1.0
#cat data_uci/docword.nips.txt | python get_docword2oneline.py | ./tfidf_cosine -d data_uci/docword.nips.txt > exp/query.out

# 3. as 2., yet using arg switch
## running sanity check to see if each doc predicts itself as the most similar, with cosine similarity as 1.0
./tfidf_cosine -d data_uci/docword.nips.txt -t data_uci/docword.nips.txt > exp/query.out
