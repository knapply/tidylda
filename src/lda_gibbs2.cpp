// Functions to make a collapsed gibbs sampler for LDA

#include <RcppArmadilloExtensions/sample.h>
#include <RcppArmadillo.h>
#define ARMA_64BIT_WORD

#include <RcppThread.h>

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
// [[Rcpp::export]]
Rcpp::List create_lexicon(arma::imat&      Cd,
                          const arma::mat& Phi,
                          arma::sp_mat&    dtm,
                          const arma::vec  alpha,
                          const bool       freeze_topics,
                          const int        threads) {
  // ***************************************************************************
  // Initialize some variables
  // ***************************************************************************

  dtm = dtm.t(); // transpose dtm to take advantage of column major & parallel
  Cd  = Cd.t();  // transpose to put columns up

  auto                    sum_alpha = sum(alpha);
  std::vector<arma::imat> docs(dtm.n_cols);
  std::vector<arma::imat> Zd(dtm.n_cols);

  auto Nk = Cd.n_rows;

  // ***************************************************************************
  // Go through each document and split it into a lexicon and then sample a
  // topic for each token within that document
  // ***************************************************************************
  RcppThread::parallelFor(
      0,
      dtm.n_cols,
      [&Cd, &Phi, &dtm, &alpha, &sum_alpha, &docs, &Zd, &Nk](unsigned int d) {
        arma::vec qz(Nk);

        // arma::ivec       topic_index = Rcpp::seq_len(Nk) - 1;
        std::vector<int> topic_index(Nk);
        std::iota(std::begin(topic_index), std::end(topic_index), 0);

        // make a temporary vector to hold token indices
        auto nd(0);

        for (std::size_t v = 0; v < dtm.n_rows; ++v) {
          nd += dtm(v, d);
        }

        arma::ivec doc(nd);
        arma::ivec zd(nd);
        arma::ivec z(1);

        // fill in with token indices
        std::size_t j = 0; // index of doc, advances when we have non-zero entries

        for (std::size_t v = 0; v < dtm.n_rows; ++v) {
          if (dtm(v, d) > 0) { // if non-zero, add elements to doc

            // calculate probability of topics based on initially-sampled Phi and Cd
            for (std::size_t k = 0; k < Nk; ++k) {
              qz[k] = Phi(k, v) * (Cd(k, d) + alpha[k]) / (nd + sum_alpha - 1);
            }

            std::size_t idx(j + dtm(v, d)); // where to stop the loop below

            while (j < idx) {
              doc[j] = v;
              z      = Rcpp::RcppArmadillo::sample(topic_index, 1, false, qz);
              zd[j]  = z[0];
              j++;
            }
          }
        }

        // fill in docs[d] with the matrix we made
        docs[d] = doc;

        Zd[d] = zd;

        RcppThread::checkUserInterrupt();
      },
      threads);

  // ***************************************************************************
  // Calculate Cd, Cv, and Ck from the sampled topics
  // ***************************************************************************
  arma::imat Cd_out(Nk, dtm.n_cols);

  Cd_out.fill(0);

  arma::ivec Ck(Nk);

  Ck.fill(0);

  arma::imat Cv(Nk, dtm.n_rows);

  Cv.fill(0);

  for (std::size_t d = 0; d < Zd.size(); ++d) {
    arma::ivec zd  = Zd[d];
    arma::ivec doc = docs[d];

    if (freeze_topics) {
      for (std::size_t n = 0; n < zd.n_elem; ++n) {
        Cd_out(zd[n], d)++;
        Ck[zd[n]]++;
      }

    } else {
      for (std::size_t n = 0; n < zd.n_elem; ++n) {
        Cd_out(zd[n], d)++;
        Ck[zd[n]]++;
        Cv(zd[n], doc[n])++;
      }
    }
  }

  // ***************************************************************************
  // Prepare output and expel it from this function
  // ***************************************************************************
  using Rcpp::_;
  return Rcpp::List::create(          //
      _["docs"] = Rcpp::wrap(docs),   //
      _["Zd"]   = Rcpp::wrap(Zd),     //
      _["Cd"]   = Rcpp::wrap(Cd_out), //
      _["Cv"]   = Rcpp::wrap(Cv),     //
      _["Ck"]   = Ck                  //
  );                                  //
}

////////////////////////////////////////////////////////////////////////////////
// Declare a bunch of voids to be called inside main sampling function
////////////////////////////////////////////////////////////////////////////////
// Functions down here are called inside of calc_lda_c()

// sample a new topic
void sample_topics(const std::vector<int>&    doc,
                   std::vector<int>&          zd,
                   arma::ivec                 z,
                   const int                  d,
                   std::vector<int>&          Ck,
                   Rcpp::IntegerMatrix&       Cd,
                   Rcpp::IntegerMatrix&       Cv,
                   const std::vector<int>&    topic_index,
                   const bool                 freeze_topics,
                   const Rcpp::NumericMatrix& Phi,
                   const std::vector<double>& alpha,
                   const Rcpp::NumericMatrix& beta,
                   const double               sum_alpha,
                   const double               sum_beta,
                   double                     phi_kv) {

  // initialize qz before loop and reset at end of each iteration
  std::vector<double> qz(topic_index.size(), 1.0);
  // for each token instance in the document
  for (std::size_t n = 0; n < doc.size(); n++) {
    // discount counts from previous run ***
    Cd(zd[n], d)--;

    // update probabilities of each topic ***
    if (freeze_topics) {
      for (std::size_t k = 0; k < qz.size(); ++k) {
        // get the correct term depending on if we freeze topics or not
        // prevent branching inside loop by when `freeze_topics` condition
        phi_kv = Phi(k, doc[n]);
        qz[k]  = phi_kv * (Cd(k, d) + alpha[k]) / (doc.size() + sum_alpha - 1);
      }

    } else {
      Cv(zd[n], doc[n])--;
      Ck[zd[n]]--;

      for (std::size_t k = 0; k < qz.size(); ++k) {
        phi_kv = (Cv(k, doc[n]) + beta(k, doc[n])) / (Ck[k] + sum_beta);
        qz[k]  = phi_kv * (Cd(k, d) + alpha[k]) / (doc.size() + sum_alpha - 1);
      }
    }

    // sample a topic ***
    z = Rcpp::RcppArmadillo::sample(topic_index, 1, false, qz);

    // update counts ***
    Cd(z[0], d)++;

    if (!freeze_topics) {
      Cv(z[0], doc[n])++;
      Ck[z[0]]++;
    }

    // record this topic for this token/doc combination
    zd[n] = z[0];

    std::fill(std::begin(qz), std::end(qz), 1.0); // reset qz before next iteration
  }                                               // end loop over each token in doc
}

// self explanatory: calculates the (log) likelihood
void fcalc_likelihood(const std::size_t          Nk,
                      const R_xlen_t             Nd,
                      const R_xlen_t             Nv,
                      const R_xlen_t             t,
                      const double               sum_beta,
                      const std::vector<int>&    Ck,
                      const Rcpp::IntegerMatrix& Cd,
                      const Rcpp::IntegerMatrix& Cv,
                      const std::vector<double>& alpha,
                      const Rcpp::NumericMatrix& beta,
                      const double               lgalpha,
                      const double               lgbeta,
                      const double               lg_alpha_len,
                      Rcpp::NumericMatrix&       log_likelihood) {
  // calculate lg_beta_count1, lg_beta_count2, lg_alph_count for this iter
  // start by zeroing them out
  auto lg_beta_count1(0.0);
  auto lg_beta_count2(0.0);
  auto lg_alpha_count(0.0);

  for (std::size_t k = 0; k < Nk; ++k) {
    lg_beta_count1 += lgamma(sum_beta + Ck[k]);

    for (R_xlen_t d = 0; d < Nd; ++d) {
      lg_alpha_count += lgamma(alpha[k] + Cd(k, d));
    }

    for (R_xlen_t v = 0; v < Nv; v++) {
      lg_beta_count2 += lgamma(beta(k, v) + Cv(k, v));
    }
  }

  lg_beta_count1 *= -1;
  log_likelihood(0, t) = t;
  log_likelihood(1, t) =
      lgalpha + lgbeta + lg_alpha_len + lg_alpha_count + lg_beta_count1 + lg_beta_count2;
}

// if user wants to optimize alpha, do that here.
// procedure likely to change similar to what Mimno does in Mallet
void foptimize_alpha(std::vector<double>&    alpha,
                     const std::vector<int>& Ck,
                     const std::size_t       sumtokens,
                     const double            sum_alpha,
                     const std::size_t       Nk) {
  constexpr double denom = 2.0;
  for (std::size_t k = 0; k < Nk; ++k) {
    alpha[k] += ((Ck[k] / static_cast<double>(sumtokens) * sum_alpha) + alpha[k]) / denom;
  }
}

// Function aggregates counts across iterations after burnin iterations
void agg_counts_post_burnin(const std::size_t          Nk,
                            const std::size_t          Nd,
                            const std::size_t          Nv,
                            const bool                 freeze_topics,
                            const Rcpp::IntegerMatrix& Cd,
                            Rcpp::IntegerMatrix&       Cd_sum,
                            const Rcpp::IntegerMatrix& Cv,
                            Rcpp::IntegerMatrix&       Cv_sum) {

  if (freeze_topics) { // split these up to prevent branching inside loop
    for (std::size_t k = 0; k < Nk; ++k) {
      for (std::size_t d = 0; d < Nd; ++d) {
        Cd_sum(k, d) += Cd(k, d);
      }
      for (std::size_t v = 0; v < Nv; ++v) {
        Cv_sum(k, v) += Cv(k, v);
      }
    }

  } else {
    for (std::size_t k = 0; k < Nk; ++k) {
      for (std::size_t d = 0; d < Nd; ++d) {
        Cd_sum(k, d) += Cd(k, d);
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
//' @param calc_likelihood bool do you want to calculate the log likelihood each
//'   iteration?
//' @param optimize_alpha bool do you want to optimize alpha each iteration?
//'
// [[Rcpp::export]]
Rcpp::List fit_lda_c(const std::vector<std::vector<int>>& docs,
                     const int                            Nk,
                     const Rcpp::NumericMatrix&           beta,
                     std::vector<double>&                 alpha,
                     Rcpp::IntegerMatrix&                 Cd,
                     Rcpp::IntegerMatrix&                 Cv,
                     std::vector<int>&                    Ck,
                     std::vector<std::vector<int>>&       Zd,
                     const Rcpp::NumericMatrix&           Phi,
                     const int                            iterations,
                     const int                            burnin,
                     const bool                           freeze_topics,
                     const bool                           calc_likelihood,
                     const bool                           optimize_alpha) {
  // ***********************************************************************
  // TODO Check quality of inputs to minimize risk of crashing the program
  // ***********************************************************************

  // ***********************************************************************
  // Variables and other set up
  // ***********************************************************************

  const auto Nv = Cv.cols();
  const auto Nd = Cd.cols();

  std::vector<double> k_alpha(alpha.size());
  std::transform(std::begin(alpha),
                 std::end(alpha),
                 std::begin(k_alpha),
                 [&Nk](double x) { return x * Nk; });

  const Rcpp::NumericMatrix v_beta = beta * Nv;
  const auto sum_alpha = std::accumulate(std::begin(alpha), std::end(alpha), 0.0);
  const auto sum_beta  = sum(beta(1, Rcpp::_));
  const auto sumtokens = std::accumulate(std::begin(Ck), std::end(Ck), 0ULL);
  auto       phi_kv(0.0);

  std::vector<int> topic_index(Nk);
  std::iota(std::begin(topic_index), std::end(topic_index), 0);

  arma::ivec z(1); // for sampling topics

  // related to burnin and averaging
  Rcpp::IntegerMatrix Cv_sum(Nk, Nv);
  Rcpp::NumericMatrix Cv_mean(Nk, Nv);
  Rcpp::IntegerMatrix Cd_sum(Nk, Nd);
  Rcpp::NumericMatrix Cd_mean(Nk, Nd);

  // related to the likelihood calculation
  Rcpp::NumericMatrix log_likelihood(2, iterations);

  auto lgbeta(0.0);       // calculated immediately below
  auto lgalpha(0.0);      // calculated immediately below
  auto lg_alpha_len(0.0); // calculated immediately below

  // indices for loops
  auto t(0);
  auto d(0);
  auto n(0);
  auto k(0);
  auto v(0);

  if (calc_likelihood &&
      !freeze_topics) { // if calc_likelihood, actually populate this stuff

    for (; n < Nv; ++n) {
      lgbeta += lgamma(beta[n]);
    }

    lgbeta = (lgbeta - lgamma(sum_beta)) * Nk; // rcpp sugar here

    for (; k < Nk; ++k) {
      lgalpha += lgamma(alpha[k]);
    }

    lgalpha = (lgalpha - lgamma(sum_alpha)) * Nd;

    for (d = 0; d < Nd; ++d) {
      lg_alpha_len += lgamma(sum_alpha + docs[d].size());
    }

    lg_alpha_len *= -1;
  }

  // ***********************************************************************
  // BEGIN ITERATIONS
  // ***********************************************************************

  for (; t < iterations; ++t) {
    // loop over documents
    for (; d < Nd; ++d) { // start loop over documents

      R_CheckUserInterrupt();

      auto doc = docs[d];

      sample_topics(doc,
                    Zd[d],
                    z,
                    d,
                    Ck,
                    Cd,
                    Cv,
                    topic_index,
                    freeze_topics,
                    Phi,
                    alpha,
                    beta,
                    sum_alpha,
                    sum_beta,
                    phi_kv);

    } // end loop over docs
    // calc likelihood ***
    if (calc_likelihood && !freeze_topics) {
      fcalc_likelihood(Nk,
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
                       log_likelihood);
    }
    // optimize alpha ***
    if (optimize_alpha && !freeze_topics) {
      foptimize_alpha(alpha, Ck, sumtokens, sum_alpha, Nk);
    }

    // aggregate counts after burnin ***
    if (burnin > -1 && t >= burnin) {
      agg_counts_post_burnin(Nk, Nd, Nv, freeze_topics, Cd, Cd_sum, Cv, Cv_sum);
    }

  } // end iterations

  // ***********************************************************************
  // Cleanup and return list
  // ***********************************************************************

  // change sum over iterations to average over iterations ***

  if (burnin > -1) {
    const double diff(iterations - burnin);

    // average over chain after burnin
    for (; k < Nk; ++k) {
      for (; d < Nd; ++d) {
        Cd_mean(k, d) = Cd_sum(k, d) / diff;
      }

      for (; v < Nv; ++v) {
        Cv_mean(k, v) = Cv_sum(k, v) / diff;
      }
    }
  }

  // Return the final list ***
  using Rcpp::_;
  return Rcpp::List::create(                //
      _["Cd"]             = Cd,             //
      _["Cv"]             = Cv,             //
      _["Ck"]             = Ck,             //
      _["Cd_mean"]        = Cd_mean,        //
      _["Cv_mean"]        = Cv_mean,        //
      _["log_likelihood"] = log_likelihood, //
      _["alpha"]          = alpha,          //
      _["beta"]           = beta            //
  );                                        //
}
