// Functions to make a collapsed gibbs sampler for LDA

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppThread)]]
#include "RcppThread.h"
#include <RcppArmadilloExtensions/sample.h>
#include <R.h>
#include <cmath>
#include <Rcpp.h>
using namespace Rcpp;

////////////////////////////////////////////////////////////////////////////////
// Declare a bunch of voids to be called inside main sampling function
////////////////////////////////////////////////////////////////////////////////
// Functions down here are called inside of calc_lda_c()

// cbind matrices that are elements of a list
IntegerMatrix cbind_list(List& x, int& rows) {
  // list of integer matrices with any number of columns and same number of rows
  
  // get number of columns
  int cols = 0;
  
  for (int j = 0; j < x.length(); j++) {
    IntegerMatrix mat = x[j];
    
    cols += mat.ncol();
  }
  
  // initialize output matrix
  IntegerMatrix out(rows, cols);
  
  // nested for loop to populate
  int col_start = 0;
  
  for (int j = 0; j < x.length(); j++) {
    IntegerMatrix mat = x[j];
    
    for (int k = 0; k < mat.cols(); k++) {
      for (int l = 0; l < rows; l ++) {
        out(l, col_start + k) = mat(l, k);
      }
    }
    col_start += mat.cols();
  }
  
  return out;
  
}

// sample a new topic
void sample_topics(
    IntegerVector& doc,
    IntegerVector& zd,
    IntegerVector& z,
    int& d,
    NumericVector& Ck,
    IntegerMatrix& Cd, 
    NumericMatrix& Cv,
    IntegerVector& topic_index,
    bool& freeze_topics,
    NumericMatrix& Phi,
    NumericVector& alpha,
    NumericMatrix& beta,
    double& sum_alpha,
    double& sum_beta,
    double& phi_kv
) {
  // for each token instance in the document
  for (int n = 0; n < doc.length(); n++) {
    
    // discount counts from previous run ***
    Cd(zd[n], d) -= 1; 
    
    
    if (! freeze_topics) {
      Cv(zd[n], doc[n]) -= 1; 
      
      Ck[zd[n]] -= 1;
    }
    
    // initialize qz here to calc on the fly
    NumericVector qz(topic_index.length());
    
    qz = qz + 1;
    
    // update probabilities of each topic ***
    for (int k = 0; k < qz.length(); k++) {
      
      // get the correct term depending on if we freeze topics or not
      if (freeze_topics) {
        phi_kv = Phi(k, doc[n]);
      } else {
        phi_kv = ((double)Cv(k, doc[n]) + beta(k, doc[n])) /
          ((double)Ck[k] + sum_beta);
      }
      
      qz[k] =  phi_kv * ((double)Cd(k, d) + alpha[k]) / 
        ((double)doc.length() + sum_alpha - 1);
      
    }
    
    // sample a topic ***
    z = RcppArmadillo::sample(topic_index, 1, false, qz);
    
    // update counts ***
    Cd(z[0], d) += 1; 
    
    if (! freeze_topics) {
      
      Cv(z[0], doc[n]) += 1; 
      
      Ck[z[0]] += 1;
      
    }
    
    // record this topic for this token/doc combination
    zd[n] = z[0];
    
  } // end loop over each token in doc
  
}

// self explanatory: calculates the (log) likelihood
void fcalc_likelihood(
    double& lg_beta_count1,
    double& lg_beta_count2,
    double& lg_alpha_count,
    int& Nk,
    int& Nd,
    int& Nv,
    int& t,
    double& sum_beta,
    NumericVector& Ck,
    IntegerMatrix& Cd,
    NumericMatrix& Cv,
    NumericVector& alpha,
    NumericVector& beta,
    double& lgalpha,
    double& lgbeta,
    double& lg_alpha_len,
    NumericMatrix& log_likelihood
) {
  
  // calculate lg_beta_count1, lg_beta_count2, lg_alph_count for this iter
  // start by zeroing them out
  lg_beta_count1 = 0.0;
  lg_beta_count2 = 0.0;
  lg_alpha_count = 0.0;
  
  for (int k = 0; k < Nk; k++) {
    
    lg_beta_count1 += lgamma(sum_beta + Ck[k]);
    
    for (int d = 0; d < Nd; d++) {
      lg_alpha_count += lgamma(alpha[k] + Cd(k, d));
    }
    
    for (int v = 0; v < Nv; v++) {
      lg_beta_count2 += lgamma(beta(k,v) + Cv(k, v));
    }
    
  }
  
  lg_beta_count1 *= -1;
  
  log_likelihood(0, t) = t;
  
  log_likelihood(1, t) = lgalpha + lgbeta + lg_alpha_len + lg_alpha_count + 
    lg_beta_count1 + lg_beta_count2;
  
}

// if user wants to optimize alpha, do that here.
// procedure likely to change similar to what Mimno does in Mallet
void foptimize_alpha(
    NumericVector& alpha, 
    NumericVector& Ck,
    int& sumtokens,
    double& sum_alpha,
    int& Nk
) {
  
  NumericVector new_alpha(Nk);
  
  for (int k = 0; k < Nk; k++) {
    
    new_alpha[k] += (double)Ck[k] / (double)sumtokens * (double)sum_alpha;
    
    new_alpha[k] += (new_alpha[k] + alpha[k]) / 2;
    
  }
  
  alpha = new_alpha;
  
}

// Function aggregates counts across iterations after burnin iterations
void agg_counts_post_burnin(
    int& Nk,
    int& Nd,
    int& Nv,
    bool& freeze_topics,
    IntegerMatrix& Cd,
    IntegerMatrix& Cd_sum,
    NumericMatrix& Cv,
    NumericMatrix& Cv_sum
) {
  for (int k = 0; k < Nk; k++) {
    for (int d = 0; d < Nd; d++) {
      
      Cd_sum(k, d) += Cd(k, d);
      
    }
    if (! freeze_topics) {
      for (int v = 0; v < Nv; v++) {
        
        Cv_sum(k, v) += Cv(k, v);
        
      }
    }
  }
}



//' Main C++ Gibbs sampler for Latent Dirichlet Allocation
//' @keywords internal
//' @description
//'   This is the C++ Gibbs sampler for LDA. "Abandon all hope, ye who enter here."
//' @param docs List with one element for each document and one entry for each token
//'   as formatted by \code{\link[tidylda]{initialize_topic_counts}}
//' @param Nk int number of topics
//' @param beta NumericMatrix for prior of tokens over topics
//' @param alpha NumericVector prior for topics over documents
//' @param Cd IntegerMatrix denoting counts of topics in documents
//' @param Cv IntegerMatrix denoting counts of tokens in topics
//' @param Ck IntegerVector denoting counts of topics across all tokens
//' @param Zd List with one element for each document and one entry for each token
//'   as formatted by \code{\link[tidylda]{initialize_topic_counts}}
//' @param Phi NumericMatrix denoting probability of tokens in topics
//' @param iterations int number of gibbs iterations to run in total
//' @param burnin int number of burn in iterations
//' @param freeze_topics bool if making predictions, set to \code{TRUE}
//' @param calc_likelihood bool do you want to calculate the log likelihood each iteration?
//' @param optimize_alpha bool do you want to optimize alpha each iteration?
// [[Rcpp::export]]
List fit_lda_c(
    const List &docs_list,
    const int &Nk,
    const int &Nd,
    const int &Nv,
    const NumericMatrix &beta,
    NumericVector alpha,
    List Cd_list,
    List Cv_list,
    List Ck_list,
    List Zd_list,
    const NumericMatrix &Phi,
    const int &iterations,
    const int &burnin,
    const bool &freeze_topics,
    const bool &calc_likelihood,
    const bool &optimize_alpha
) {
  
  // ***********************************************************************
  // Variables and other set up
  // ***********************************************************************
  
  NumericMatrix Cv = Cv_list[1]; // global Cv count matrix
  
  NumericVector Ck = Ck_list[1]; // global Ck count vector
  
  IntegerMatrix Cd(Nk, Nd);
  
  NumericVector k_alpha = alpha * Nk;
  
  NumericMatrix v_beta = beta * Nv;
  
  double sum_alpha = sum(alpha);
  
  double sum_beta = sum(beta(1, _));
  
  int sumtokens = sum(Ck);
  
  double phi_kv(0.0);
  
  IntegerVector topic_index = seq_len(Nk) - 1;

  // related to burnin and averaging
  NumericMatrix Cv_sum(Nk, Nv);
  
  NumericMatrix Cv_mean(Nk, Nv);
  
  IntegerMatrix Cd_sum(Nk, Nd);
  
  NumericMatrix Cd_mean(Nk, Nd);
  
  // related to the likelihood calculation
  NumericMatrix log_likelihood(2, iterations);
  
  double lgbeta(0.0); // calculated immediately below
  
  double lgalpha(0.0); // calculated immediately below
  
  double lg_alpha_len(0.0); // calculated immediately below
  
  double lg_beta_count1(0.0); // calculated at the bottom of the iteration loop
  
  double lg_beta_count2(0.0); // calculated at the bottom of the iteration loop
  
  double lg_alpha_count(0.0); // calculated at the bottom of the iteration loop
  
  if (calc_likelihood && ! freeze_topics) { // if calc_likelihood, actually populate this stuff
    
    for (int n = 0; n < Nv; n++) {
      lgbeta += lgamma(beta[n]);
    }
    
    lgbeta = (lgbeta - lgamma(sum_beta)) * Nk; // rcpp sugar here
    
    for (int k = 0; k < Nk; k++) {
      lgalpha += lgamma(alpha[k]);
    }
    
    lgalpha = (lgalpha - lgamma(sum_alpha)) * Nd;
    
    for (int j = 0; j < docs_list.length(); j++) {
      
      List docs = docs_list[j];
      
      for (int d = 0; d < Nd; d++) {
        IntegerVector doc = docs[d];
        
        lg_alpha_len += lgamma(sum_alpha + doc.length());
      }
    }
    

    
    lg_alpha_len *= -1;
  }
  
  
  
  // ***********************************************************************
  // BEGIN ITERATIONS
  // ***********************************************************************
  
  for (int t = 0; t < iterations; t++) {
    
    //////////////////////////////
    // parallel loop over batches
    //////////////////////////////
    
    // int num_threads = Cd_list.length(); // number of parallel threads
    // 
    // RcppThread::parallelFor(
    //   0, docs_list.length() - 1,
    //   [
    // &docs_list,
    // &Zd_list,
    // &Cd_list,
    // &Ck_list,
    // &Cv_list,
    // &topic_index, // collision risk?
    // &freeze_topics,
    // &Phi,
    // &alpha,
    // &beta,
    // &sum_alpha,
    // &sum_beta,
    // &phi_kv
    //   ](int j){
    // 
    //     List docs = docs_list[j];
    // 
    //     IntegerMatrix Cd = Cd_list[j];
    // 
    //     NumericMatrix Cv = Cv_list[j];
    // 
    //     NumericVector Ck = Ck_list[j];
    // 
    //     List Zd = Zd_list[j];
    // 
    //     IntegerVector z(1); // for sampling topics
    // 
    //       // loop over documents
    //       for (int d = 0; d < Zd.length(); d++) { //start loop over documents
    // 
    //         RcppThread::checkUserInterrupt();
    // 
    //         IntegerVector doc = docs[d];
    // 
    //         IntegerVector zd = Zd[d];
    // 
    //         sample_topics(
    //           doc,
    //           zd,
    //           z,
    //           d,
    //           Ck,
    //           Cd,
    //           Cv,
    //           topic_index,
    //           freeze_topics,
    //           Phi,
    //           alpha,
    //           beta,
    //           sum_alpha,
    //           sum_beta,
    //           phi_kv
    //         );
    // 
    //       } // end loop over docs
    // 
    //   },
    //   num_threads
    // ); // end loop over batches
    
    
    // for troubleshooting purposes

    Rcout << "iteration " << t << "\n";

    for (int j = 0; j < Zd_list.length(); j++) {

      Rcout << "looping over batches\n";

      List docs = docs_list[j];

      IntegerMatrix Cd_batch = Cd_list[j];

      NumericMatrix Cv_batch = Cv_list[j];

      NumericVector Ck_batch = Ck_list[j];

      List Zd = Zd_list[j];

      IntegerVector z(1); // for sampling topics

      // loop over documents
      for (int d = 0; d < Zd.length(); d++) { //start loop over documents

        // Rcout << "doc " << d << "\n";

        R_CheckUserInterrupt();

        IntegerVector doc = docs[d];

        IntegerVector zd = Zd[d];

        sample_topics(
          doc,
          zd,
          z,
          d,
          Ck_batch,
          Cd_batch,
          Cv_batch,
          topic_index,
          freeze_topics,
          Phi,
          alpha,
          beta,
          sum_alpha,
          sum_beta,
          phi_kv
        );

      } // end loop over docs

      // Rcout << "end document loops\n";
    }
    
    //////////////////////////////
    // Reconcile Cv etc. across batches
    //////////////////////////////
    
    Rcout << "Reconcile Cv and Ck\n";
    
    Cv = Cv * 0.0;
    
    Ck = Ck * 0.0;
    
    for (int j = 0; j < Cv_list.length(); j++) {
      
      // Global Cv is average across batches
      NumericMatrix tmp_m = Cv_list[j];
      
      for (int cols = 0; cols < tmp_m.ncol(); cols++) {
        for (int rows = 0; rows < tmp_m.nrow(); rows++){
          Cv(rows, cols) = (Cv(rows, cols) + tmp_m(rows, cols));
        }
      }
      
      // Cv = Cv / (double)Cv_list.length();
      
      // Global Ck is average across batches
      NumericVector tmp_v = Ck_list[j];
      
      for (int k = 0; k < tmp_v.length(); k++) {
        Ck[k] = (Ck[k] + tmp_v[k]);
      }
      
      // Ck = Ck / (double)Cv_list.length();
      
    }


    Rcout << "sum(Ck) is " << sum(Ck) << "\n";
    
    // Global Cd is cbind() of batch Cd's
    Rcout << "combind global Cd\n";
    
    Cd = cbind_list(Cd_list, Nk);

    //////////////////////////////
    // Do other calculations
    //////////////////////////////
    
    // calc likelihood ***
    if (calc_likelihood && ! freeze_topics) {
      
      fcalc_likelihood(
        lg_beta_count1,
        lg_beta_count2,
        lg_alpha_count,
        Nk,
        Nd,
        Nv,
        t,
        sum_beta,
        Ck,
        Cd,
        Cv,
        alpha,
        beta,
        lgalpha,
        lgbeta,
        lg_alpha_len,
        log_likelihood
      );
      
    }
    // optimize alpha ***
    if (optimize_alpha && ! freeze_topics) {
      
      foptimize_alpha(
        alpha, 
        Ck,
        sumtokens,
        sum_alpha,
        Nk
      );  
      
    }
    
    // aggregate counts after burnin ***
    if (burnin > -1 && t >= burnin) {
      
      agg_counts_post_burnin(
        Nk,
        Nd,
        Nv,
        freeze_topics,
        Cd,
        Cd_sum,
        Cv,
        Cv_sum
      );
      
    }
    
    
    // copy global Cv and Ck back to batches
    Rcout << "copy global Cv and Ck back to batches\n";
    
    for (int j = 0; j < Cv_list.length(); j++) {
      
      Cv_list[j] = clone(Cv);
      
      Ck_list[j] = clone(Ck);
      
    }
    
  } // end iterations
  
  // ***********************************************************************
  // Cleanup and return list
  // ***********************************************************************
  
  // change sum over iterations to average over iterations ***
  
  if (burnin >-1) {
    
    double diff = iterations - burnin;
    
    // average over chain after burnin 
    for (int k = 0; k < Nk; k++) {
      
      for (int d = 0; d < Nd; d++) {
        Cd_mean(k, d) = ((double)Cd_sum(k, d) / diff);
      }
      
      for (int v = 0; v < Nv; v++) {
        Cv_mean(k, v) = ((double)Cv_sum(k, v) / diff);
      }
    }
  }
  
  // Return the final list ***
  
  return List::create(
    Named("Cd") = Cd,
    Named("Cv") = Cv,
    Named("Ck") = Ck,
    Named("Cd_mean") = Cd_mean,
    Named("Cv_mean") = Cv_mean,
    Named("log_likelihood") = log_likelihood,
    Named("alpha") = alpha,
    Named("beta") = beta
  );  
}


