tfidf_cosine_cpp
================

Yang Gao's implementation of tf-idf text indexing scheme, predict doc similarity by cosine similarity. Refer to: http://en.wikipedia.org/wiki/Tf–idf; yet I use normalized tf instead of raw tf.

It can serve as a baseline for more complicated text indexing and retrieval models, such as topic model.

usage
=====
-  see "run_examples.sh" for example usage.

dependencies
============
1. external libraries, such as Eigen and tclap, are included; therefore the code is ready to run

compiling
=========
1. for initial build, type "make";
2. if you modify code, type "make rebuild"

questions
=========
for questions, comments or to report bugs, contact Yang Gao(USC/ISI) at yanggao1119@gmail.com
