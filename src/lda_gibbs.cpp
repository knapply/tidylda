



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
    NumericVector& Ck,
    IntegerVector& Cd_doc, 
    NumericMatrix& Cv,
    IntegerVector& topic_index,
    bool& freeze_topics,
    // NumericMatrix& Phi,
    NumericVector& alpha,
    NumericMatrix& beta,
    double& sum_alpha,
    double& sum_beta
) {
  IntegerVector z = 0;
  
  // for each token instance in the document
  for (int n = 0; n < doc.length(); n++) {
    
    // discount counts from previous run ***
    Cd_doc[zd[n]] -= 1; 
    
    
    if (! freeze_topics) {
      Cv(zd[n], doc[n]) -= 1; 
      
      Ck[zd[n]] -= 1;
    }
    
    // Rcout << "zd[n]: " << zd[n] << " token: " << doc[n] << "\n";
    
    // initialize qz here to calc on the fly
    NumericVector qz(topic_index.length());
    
    qz = qz + 1;
    
    // update probabilities of each topic ***
    for (int k = 0; k < qz.length(); k++) {
      
      // get the correct term depending on if we freeze topics or not
      double phi_kv = 0.0; 
      
      // if (freeze_topics) {
      //   phi_kv = Phi(k, doc[n]);
      // } else {
      //   phi_kv = ((double)Cv(k, doc[n]) + beta(k, doc[n])) /
      //     ((double)Ck[k] + sum_beta);
      // }
      
      phi_kv = ((double)Cv(k, doc[n]) + beta(k, doc[n])) /
        ((double)Ck[k] + sum_beta);
      
      qz[k] =  phi_kv * ((double)Cd_doc[k] + alpha[k]) / 
        ((double)doc.length() + sum_alpha - 1);
      
    }
    
    // sample a topic ***
    z = RcppArmadillo::sample(topic_index, 1, false, qz);
    
    // update counts ***
    Cd_doc[z[0]] += 1; 
    
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
    List& batches,
    List& docs,
    List Zd,
    IntegerMatrix Cd,
    NumericMatrix Cv,
    NumericVector Ck,
    NumericVector alpha,
    NumericMatrix& beta,
    int& iterations,
    int& burnin,
    bool& freeze_topics,
    bool& calc_likelihood,
    bool& optimize_alpha //, NumericMatrix& Phi
) {
  
  Rcout << "Num docs " << Cd.ncol() << "\n";
  
  
  // ***********************************************************************
  // Variables and other set up
  // ***********************************************************************
  int Nk = Ck.length();
  
  int Nd = Cd.ncol();
  
  int Nv = Cv.ncol();
  
  int num_batches = batches.length();
  
  NumericVector k_alpha = alpha * Nk;
  
  NumericMatrix v_beta = beta * Nv;
  
  double sum_alpha = sum(alpha);
  
  double sum_beta = sum(beta(1, _)); // this assumes magnitude of each row
  // of beta is the same. Will fail when
  // you integrate the seeding topics
  // functionality.
  
  int sumtokens = sum(Ck);
  
  IntegerVector topic_index = seq_len(Nk) - 1;
  
  // related to averaging after burnin
  NumericMatrix Cv_sum(Nk, Nv);
  
  IntegerMatrix Cd_sum(Nk, Nd);
  
  
  // ***********************************************************************
  // Initialize objects for likelihood calculation  
  // ***********************************************************************
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
    
    for (int d = 0; d < Nd; d++) {
      IntegerVector doc = docs[d];
      
      lg_alpha_len += lgamma(sum_alpha + doc.length());
    }
    
    lg_alpha_len *= -1;
  }
  
  
  // ***********************************************************************
  // BEGIN ITERATIONS
  // ***********************************************************************
  
  int t;
  
  for (t = 0; t < iterations; t++) {
    
    ////////////////////////////////////////////////////////////
    // get Cv and Ck for each batch
    ////////////////////////////////////////////////////////////
    List Cv_batches(num_batches);
    
    List Ck_batches(num_batches);
    
    for (int j = 0; j < num_batches; j++) {
      
      Cv_batches[j] = clone(Cv);
      
      Ck_batches[j] = clone(Ck);
      
    }
    
    
    ////////////////////////////////////////////////////////////
    // parallel loop over batches
    ////////////////////////////////////////////////////////////
    
    for (int j = 0; j < num_batches; j++) {
      
      IntegerVector batch_idx = batches[j];
      
      NumericMatrix Cv_batch = Cv_batches[j];
      
      NumericVector Ck_batch = Ck_batches[j];
      
      // loop over documents in that batch
      for (int d = batch_idx[0]; d < batch_idx[batch_idx.length()]; d++) {
        // Rcout << "document " << d << "\n";
        
        R_CheckUserInterrupt();
        
        IntegerVector doc = docs[d];
        
        IntegerVector zd = Zd[d];
        
        IntegerVector Cd_doc = Cd(_, d);
        
        sample_topics(
          doc,
          zd,
          Ck_batch,
          Cd_doc, 
          Cv_batch,
          topic_index,
          freeze_topics,
          // Phi,
          alpha,
          beta,
          sum_alpha,
          sum_beta
        );
        
      } // end loop over docs
      
    } // end loop over batches
    
    Rcout << "iteration " << t << "\n";
    
    ////////////////////////////////////////////////////////////
    // Reconcile Cv and Ck across batches
    ////////////////////////////////////////////////////////////
    Cv = Cv * 0.0;
    
    Ck = Ck * 0.0;
    
    for (int j = 0; j < num_batches; j++) {
      
      // Global Cv is average across batches
      NumericMatrix mat = Cv_batches[j];
      
      Cv += (mat / (double)num_batches);
      
      // Global Ck is average across batches
      NumericVector vec = Ck_batches[j];
      
      Ck += (vec / (double)num_batches);
      
    }
    
    ////////////////////////////////////////////////////////////
    // Additional things: likelihood, alpha, aggregate
    ////////////////////////////////////////////////////////////
    
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
    
  } // finish iterations
  
  
  // ***********************************************************************
  // Cleanup and return list
  // ***********************************************************************
  
  NumericMatrix Cv_mean(Nk, Nv);
  
  NumericMatrix Cd_mean(Nk, Nd);
  
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
    Named("Cv_mean") = Cv_sum, // Cv_mean,
    Named("log_likelihood") = log_likelihood,
    Named("alpha") = alpha,
    Named("beta") = beta
  );  
  
}

/*** R
source(paste0(here::here(), "/R/utils.R"))

dtm <- textmineR::nih_sample_dtm
k <- 10
alpha <- 0.1
beta = 0.01
alpha <- tidylda:::format_alpha(alpha = alpha, k = k)
beta <- tidylda:::format_beta(beta = beta, k = k, Nv = ncol(dtm))

counts <- tidylda:::initialize_topic_counts(
  dtm = dtm, k = k,
  alpha = alpha$alpha, beta = beta$beta
)

# divide into batches to enable parallel execution of the Gibbs sampler
batch_indices <- count_bridge(
  lda_threads = 3,
  num_docs = nrow(dtm)
)

### run C++ gibbs sampler ----

lda <- fit_lda_c(
  batches = batch_indices,
  docs = counts$docs,
  Zd = counts$Zd,
  Cd = counts$Cd,
  Cv = counts$Cv,
  Ck = counts$Ck,
  alpha = alpha$alpha,
  beta = beta$beta,
  iterations = 10,
  burnin = 5,
  freeze_topics = TRUE,
  calc_likelihood = TRUE,
  optimize_alpha = TRUE
)
*/
  