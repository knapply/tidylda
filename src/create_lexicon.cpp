// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadilloExtensions/sample.h>
#include <R.h>
#include <cmath>
#include <Rcpp.h>
#define ARMA_64BIT_WORD
using namespace Rcpp;

//' Make a lexicon for looping over in the gibbs sampler
//' @keywords internal
//' @description
//'   One run of the Gibbs sampler and other magic to initialize some objects.
//'   Works in concert with \code{\link[tidylda]{initialize_topic_counts}}.
//' @param Cd IntegerMatrix denoting counts of topics in documents
//' @param Phi NumericMatrix denoting probability of words in topics
//' @param dtm arma::sp_mat document term matrix
//' @param alpha NumericVector prior for topics over documents
//' @param freeze_topics bool if making predictions, set to \code{TRUE}
//[[Rcpp::export]]
List create_lexicon(
    IntegerMatrix &Cd, 
    NumericMatrix &Phi, 
    arma::sp_mat &dtm,
    NumericVector alpha,
    bool freeze_topics
) {
  
  // ***************************************************************************
  // Initialize some variables
  // ***************************************************************************
  
  double sum_alpha = sum(alpha);
  
  List docs(dtm.n_rows); 
  
  List Zd(dtm.n_rows);
  
  int Nk = Cd.nrow();
  
  NumericVector qz(Nk);
  
  IntegerVector topic_index = seq_len(Nk) - 1;
  
  // ***************************************************************************
  // Go through each document and split it into a lexicon and then sample a 
  // topic for each token within that document
  // ***************************************************************************
  for (int d = 0; d < dtm.n_rows; d++) {
    
    // make a temporary vector to hold token indices
    int nd = 0;
    
    for (int v = 0; v < dtm.n_cols; v++) {
      nd += dtm(d,v);
    }
    
    IntegerVector doc(nd);
    
    IntegerVector zd(nd);
    
    IntegerVector z(1);
    
    // fill in with token indices
    int j = 0; // index of doc, advances when we have non-zero entries 
    
    for (int v = 0; v < dtm.n_cols; v++) {
      
      if (dtm(d,v) > 0) { // if non-zero, add elements to doc
        
        // calculate probability of topics based on initially-sampled Phi and Cd
        for (int k = 0; k < Nk; k++) {
          qz[k] = Phi(k, v) * ((double)Cd(k, d) + alpha[k]) / ((double)nd + sum_alpha - 1);
        }
        
        int idx = j + dtm(d,v); // where to stop the loop below
        
        while (j < idx) {
          
          doc[j] = v;
          
          z = RcppArmadillo::sample(topic_index, 1, false, qz);
          
          zd[j] = z[0];
          
          j += 1;
        }
        
      }
    }
    
    // fill in docs[d] with the matrix we made
    docs[d] = doc;
    
    Zd[d] = zd;
    
    R_CheckUserInterrupt();
    
  }
  
  // ***************************************************************************
  // Calculate Cd, Cv, and Ck from the sampled topics
  // ***************************************************************************
  IntegerMatrix Cd_out(Nk, dtm.n_rows);
  
  IntegerVector Ck(Nk);
  
  IntegerMatrix Cv(Nk, dtm.n_cols);
  
  for (int d = 0; d < Zd.length(); d++) {
    
    IntegerVector zd = Zd[d]; 
    
    IntegerVector doc = docs[d];
    
    for (int n = 0; n < zd.length(); n++) {
      
      Cd_out(zd[n], d) += 1;
      
      Ck[zd[n]] += 1;
      
      if (! freeze_topics) {
        Cv(zd[n], doc[n]) += 1;
      }
      
    } 
    
  }
  
  // ***************************************************************************
  // Prepare output and expel it from this function
  // ***************************************************************************
  
  return List::create(
    Named("docs") = docs,
    Named("Zd") = Zd,
    Named("Cd") = Cd_out,
    Named("Cv") = Cv,
    Named("Ck") = Ck
  );
  
}
