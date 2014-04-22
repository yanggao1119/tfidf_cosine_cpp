CXX=g++
EIGEN=eigen-eigen-6b38706d90a9
TCLAP=tclap-1.2.1/include
#DEIGEN_NO_DEBUG is imperative for good performance
CFLAGS=-DEIGEN_DONT_PARALLELIZE -DEIGEN_NO_DEBUG -DNDEBUG -O3 

SOURCE=tfidf_cosine.cpp
EXE=tfidf_cosine

#Setting the Architecture
all:
	$(CXX) $(SOURCE) -I$(EIGEN) -I$(TCLAP) $(CFLAGS) -o $(EXE)

clean:
	rm -rf $(EXE)

rebuild: clean all
