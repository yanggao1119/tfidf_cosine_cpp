#include <vector>
#include <map>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tclap/CmdLine.h>

using namespace std;
using namespace Eigen;
using namespace TCLAP;


// pair comparer copied from shaofeng mo
bool pairCmpDescend( const pair<int,double>& p1, const pair<int,double>& p2){
  if( p1.second - p2.second > 0.00000000001 ){
    return true;
  }else{
    return false;
  }
}


bool pairCmpAscend( const pair<int,double>& p1, const pair<int,double>& p2){
  if( p1.second - p2.second < 0.00000000001 ){
    return true;
  }else{
    return false;
  }
}


struct Document {
    vector<int> word_ind;
    vector<int> word_count;
};


Document* get_str2doc(string str) 
{
    // create Document struct from one-line str, return pointer
    Document * doc = new Document;
	//NOTE: istringstream is ambiguous with tclap; compiling error w/o std scoping
    std::istringstream iss(str);
    int w_i, w_c; 
    while (iss >> w_i >> w_c) 
    {
        //NOTE: convert 1-based word and doc index to be internally 0-based
        w_i -= 1;
        doc->word_ind.push_back(w_i);
        doc->word_count.push_back(w_c);
    }
    /*for (int i=0; i<doc->word_ind.size(); i++)
    {
        cerr << "w_i:" << doc->word_ind[i] << "\tw_c:" << doc->word_count[i] << endl;
        
    }*/
    return doc;
}


void read_vocab(string file, vector<string> & vocabs) 
{
    clock_t t_b4_read = clock();

    ifstream ifs(file.c_str());
    for (string line ; getline(ifs, line); )
    {
       vocabs.push_back(line); 
    }
    cerr << "done reading vocab file, time spent: " << float(clock() - t_b4_read)/CLOCKS_PER_SEC << " secs" << endl << endl;
}


Document** read_docword(string file, int & D, int & W, int & C) 
{
    clock_t t_b4_read = clock();

    Document** documents;
    ifstream ifs(file.c_str());
    int line_count = 0;
    for (string line ; getline(ifs, line); )
    {
        //cerr << "line*"<<line<<"*\n";
        line_count += 1;
        if (line_count % 100000 == 0) cerr << "reading " << line_count << " lines" << endl;
        if (line_count == 1) 
        {
            D = atoi(line.c_str());
            documents = new Document* [D];
            for (int counter = 0; counter < D; counter++)
            {
                documents[counter] = new Document;
            }
        }
        else if (line_count == 2)
        {
            W = atoi(line.c_str());
        }
        else if (line_count != 3) 
        {
			//NOTE: istringstream is ambiguous with tclap; compiling error w/o std scoping
            std::istringstream iss(line);
            int d_j, w_i, w_c;
            iss >> d_j >> w_i >> w_c;
            C += w_c;
            //NOTE: convert 1-based word and doc index to be internally 0-based
            d_j -= 1;
            w_i -= 1;
            //cerr << d_j << " " << w_i << " " << w_c << endl;
            documents[d_j]->word_ind.push_back(w_i);
            documents[d_j]->word_count.push_back(w_c);
        }
    }
    cerr << "D:" << D << "\tW:" << W << "\tC:" << C << endl; 
    cerr << "done reading docword file, time spent: " << float(clock() - t_b4_read)/CLOCKS_PER_SEC << " secs" << endl << endl;
    return documents;
}


void get_tfidf_test(MatrixXd & mat_tfidf_test,
                    Document * document,
                    map<int, double> & word_ind2idf,
                    const int & W)
{
    // return 1 by W tfidf matrix for test doc
    mat_tfidf_test = MatrixXd::Constant(1, W, 0);  

    // compute doc length
    double C_j = 0.0;
    for (int i=0;i<document->word_count.size();i++)
    {
        C_j += document->word_count[i];
    }

    // populate mat_tfidf_test
    for (int i=0; i<document->word_ind.size(); i++)
    {
        int w_i = document->word_ind[i];
        mat_tfidf_test(0, w_i) = ( document->word_count[i] / C_j )* word_ind2idf[w_i]; 
    }
    cerr << "tfidf mat for test" << endl << mat_tfidf_test.row(0) << endl;
}


void get_tfidf_train(MatrixXd & mat_tfidf_train,
                        Document ** documents,
                        map<int, double> & word_ind2idf,
                        const int & D,
                        const int & W)
{
    // this function has two purposes:
    // 1. return D by W tfidf matrix for train docs
    // 2. return word_ind2idf hashmap for test doc lookup
    mat_tfidf_train = MatrixXd::Constant(D, W, 0);  

    //NOTE: having total vocab len W, further assume that
    for (int w_i=0; w_i < W; w_i++)
    {
        word_ind2idf[w_i] = 0;
    }
    // accumulate doc counts
    for (int j=0; j<D; j++)
    {
        for (int i=0; i<documents[j]->word_ind.size(); i++)
        {
            int w_i = documents[j]->word_ind[i];
            word_ind2idf[w_i] += 1;
        }
    }
    // convert to idf
    for (int w_i=0; w_i < W; w_i++)
    {
        word_ind2idf[w_i] = log10( D * 1.0 / word_ind2idf[w_i] );
    }

    // populate mat_tfidf_train
    for (int j=0; j<D; j++)
    {
        // compute doc length
        double C_j = 0.0;
        for (int i=0;i<documents[j]->word_count.size();i++)
        {
            C_j += documents[j]->word_count[i];
        }

        for (int i=0; i<documents[j]->word_ind.size(); i++)
        {
            int w_i = documents[j]->word_ind[i];
            mat_tfidf_train(j, w_i) = ( documents[j]->word_count[i] / C_j ) * word_ind2idf[w_i]; 
        }
        cerr << "tfidf mat for train " << j << endl << mat_tfidf_train.row(j) << endl;
    }
}


void get_similar_by_cosine_similarity(vector< pair<int, double> > & pair_docind_score,
                                        const MatrixXd & mat_tfidf_train,
                                        const MatrixXd & mat_tfidf_test,
                                        const int & D,
                                        const int & W)

{
    // given testdoc, rank training doc by cosine similarity: higher cosine similarity, higher rank
    double norm_testvec = mat_tfidf_test.row(0).norm();
    for (int j=0; j<D; j++)
    {
        double dist = mat_tfidf_train.row(j).dot( mat_tfidf_test.row(0) ) / norm_testvec / mat_tfidf_train.row(j).norm();
        pair_docind_score.push_back( pair<int, double> (j, dist) );
    }

    sort( pair_docind_score.begin() , pair_docind_score.end() , pairCmpDescend );
}


int main( int argc,      // Number of strings in array argv
          char *argv[],   // Array of command-line argument strings
          char *envp[] )  // Array of environment variable strings
{
    //speed up c++ io stream
    std::ios::sync_with_stdio(false);

    try {  
        CmdLine cmd("Command description message", ' ', "0.2");

        ValueArg<string> docwordInFileArg("d","docwordfile","Path to docword file for training, in uci sparse bag-of-words format, where each line is a (doc id, word id, word count) triple delimited by space. Both doc id and word id are 1-based. Word id refers to the corresponding line in the vocab file", true, "","string");
        cmd.add( docwordInFileArg );

        //TODO: output vocab for better visualization and sanity check, optional
        ValueArg<string> vocabInFileArg("v","vocabfile","Path to vocab file associated with docword file for training, in uci sparse bag-of-words format, 1-based", false, "","string");
        cmd.add( vocabInFileArg );

        ValueArg<string> docwordTestInFileArg("t","docwordtestfile","Path to docword file for similarity testing, in uci sparse bag-of-words format. Output similarity prediction for each file to STDOUT. Note that test file can also be piped from STDIN in a compact oneline format, yet this arg switch is preferred to STDIN when there are empty lines", false, "","string");
        cmd.add( docwordTestInFileArg );

        ValueArg<int> similarSizeArg("","similarsize","for similarity test, how many top similar docs to report", false, 50,"int");
        cmd.add( similarSizeArg );

        //SwitchArg predictSimilarSwitch("p","predict","optional prediction mode, i.e., after training, it accepts a one-line compact docword representation from STDIN and outputs to STDOUT the id of the most similar document (1-based) from the training set", false);
        //cmd.add( predictSimilarSwitch );

        cmd.parse( argc, argv );

        // Get the value parsed by each arg. 
        const string f_docword = docwordInFileArg.getValue();
        const string f_docword_test = docwordTestInFileArg.getValue();
        const string f_vocab = vocabInFileArg.getValue();
        const int SIMILAR_SIZE = similarSizeArg.getValue();

        //const bool predictSimilar = predictSimilarSwitch.getValue();

        // report parameters
        cerr << "\nParameters:\n";
        cerr << "Input train docword file: " << f_docword << endl;
        cerr << "Input train vocab file: " << f_vocab << endl;
        cerr << "Input test docword file: " << f_docword_test << endl;
        cerr << "SIMILAR_SIZE: " << SIMILAR_SIZE << endl;
        cerr << endl;

        // read vocab file
        vector<string> vocabs;
        read_vocab(f_vocab, vocabs);
        //for (int i=0; i<vocabs.size(); i++) cerr << vocabs[i] << endl;
        //cerr << "total vocab " << vocabs.size() << endl;

        // read docword file and obtain D, W, C
        int D=0, W=0, C=0;
        //NOTE: error if we pass this pointer as function arg
        //Document** documents = NULL;
        Document** documents = read_docword(f_docword, D, W, C);
        //TODO: assert to check size consistency of docword file and vocab file
        //assert (vocabs.size()==W);
  
        // report empty doc to stdout, convert internal 0-based doc ind to 1-based
        int empty_doc_count = 0;
        for (int d=0; d<D; d++)
        {
            if (documents[d]->word_ind.size() == 0)
            { 
                cerr << "empty doc: " << d+1 << endl;
                empty_doc_count++;
            }
        }
        cerr << "# of empty doc: " << empty_doc_count << endl << endl;

        // tf-idf
        // bookkeeping word_ind2idf for test case
        map<int, double> word_ind2idf;
        // create tfidf matrix for dot product in cosine similarity
        MatrixXd mat_tfidf_train;
        get_tfidf_train(mat_tfidf_train, documents, word_ind2idf, D, W);

        // similarity testing based on cosine similarity, input from STDIN, output to STDOUT
        //if (predictSimilar)
        if (true)
        {
            cerr << "\nReading input from STDIN" << endl;
            while (cin) 
            {
                string testdoc_line;
                getline(cin, testdoc_line);
                if (!cin.eof())
                {
                    clock_t t_b4_test = clock();
                    Document * testdoc = get_str2doc(testdoc_line);
                    MatrixXd mat_tfidf_test;
                    get_tfidf_test(mat_tfidf_test, testdoc, word_ind2idf, W);

                    // get paired (doc_ind, score) vector with similarity measure
                    vector< pair<int, double> > pair_docind_score;
                    get_similar_by_cosine_similarity(pair_docind_score, mat_tfidf_train, mat_tfidf_test, D, W);

                    // output to STDOUT in one line
                    int numTopSimilar = (pair_docind_score.size() < SIMILAR_SIZE ? pair_docind_score.size() : SIMILAR_SIZE);
                    for (int p=0; p<numTopSimilar; p++)
                    {
                        cout << pair_docind_score[p].first+1 << ":" << pair_docind_score[p].second;
                        if (p < numTopSimilar-1) cout << " ";
                    }
                    cout << endl;
                    cerr << "time spent: " << float(clock() - t_b4_test)/CLOCKS_PER_SEC << " secs" << endl; 
                    cout << endl;
                }
            }
        }

    } catch (ArgException &e)  // catch any exceptions
    { cerr << "error: " << e.error() << " for arg " << e.argId() << endl; }

}
