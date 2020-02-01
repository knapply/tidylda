#' Fit a Latent Dirichlet Allocation topic model
#' @description Fit a Latent Dirichlet Allocation topic model using collapsed Gibbs sampling. 
#' @param dtm A document term matrix or term co-occurrence matrix of class dgCMatrix.
#' @param k Integer number of topics.
#' @param iterations Integer number of iterations for the Gibbs sampler to run. 
#' @param burnin Integer number of burnin iterations. If \code{burnin} is greater than -1,
#'        the resulting "phi" and "theta" matrices are an average over all iterations
#'        greater than \code{burnin}.
#' @param alpha Numeric scalar or vector of length \code{k}. This is the prior 
#'        for topics over documents.
#' @param beta Numeric scalar, numeric vector of length \code{ncol(dtm)}, 
#'        or numeric matrix with \code{k} rows and \code{ncol(dtm)} columns.
#'        This is the prior for words over topics.
#' @param optimize_alpha Logical. Do you want to optimize alpha every iteration?
#'        Defaults to \code{FALSE}. See 'details' below for more information.
#' @param calc_likelihood Logical. Do you want to calculate the log likelihood every iteration?
#'        Useful for assessing convergence. Defaults to \code{FALSE}. 
#' @param calc_r2 Logical. Do you want to calculate R-squared after the model is trained?
#'        Defaults to \code{FALSE}. This calls \code{\link[textmineR]{CalcTopicModelR2}}.
#' @param return_data Logical. Do you want \code{dtm} returned as part of the model object?
#' @param ... Other arguments to be passed to \code{\link[furrr]{future_map}}
#' @return Returns an S3 object of class \code{tidylda}. 
#' @details This function calls a collapsed Gibbs sampler for Latent Dirichlet Allocation
#'   written using the excellent Rcpp package. Some implementation notes follow:
#'   
#'   When you use burn-in iterations (i.e. \code{burnin = TRUE}), the resulting 
#'   \code{phi} and \code{theta} matrices are calculated by averaging over every 
#'   iteration after the specified  number of burn-in iterations. If you do not 
#'   use burn-in iterations, then the matrices are calculated from the last run
#'   only. Ideally, you'd burn in every iteration before convergence, then average
#'   over the chain after its converved (and thus every observation is independent).
#'   
#'   If you set \code{optimize_alpha} to \code{TRUE}, then each element of \code{alpha}
#'   is proportional to the number of times each topic has be sampled that iteration
#'   averaged with the value of \code{alpha} from the previous iteration. This lets
#'   you start with a symmetric \code{alpha} and drift into an asymmetric one. 
#'   However, (a) this probably means that convergence will take longer to happen 
#'   or convergence may not happen at all. And (b) I make no guarantees that doing this
#'   will give you any benefit or that it won't hurt your model. Caveat emptor!
#'   
#'   The log likelihood calculation is the same that can be found on page 9 of
#'   \url{https://arxiv.org/pdf/1510.08628.pdf}. The only difference is that the
#'   version in \code{\link[tidylda]{tidylda}} allows \code{beta} to be a
#'   vector or matrix. (Vector used in this function, matrix used for model
#'   updates in \code{\link[tidylda]{update.tidylda}}. At present, the 
#'   log likelihood function appears to be ok for assessing convergence. i.e. It 
#'   has the right shape. However, it is, as of this writing, returning positive
#'   numbers, rather than the expected negative numbers. Looking into that, but 
#'   in the meantime caveat emptor once again.
#'   
#' @examples 
#' # load some data
#' data(nih_sample_dtm, package = "textmineR")
#' 
#' # fit a model 
#' set.seed(12345)
#' m <- tidylda(dtm = nih_sample_dtm[1:20,], k = 5,
#'                  iterations = 200, burnin = 175)
#'
#' str(m)
#' 
#' # predict on held-out documents using gibbs sampling "fold in"
#' p1 <- predict(m, nih_sample_dtm[21:100,], method = "gibbs",
#'               iterations = 200, burnin = 175)
#' 
#' # predict on held-out documents using the dot product method
#' p2 <- predict(m, nih_sample_dtm[21:100,], method = "dot")
#'
#' # compare the methods
#' barplot(rbind(p1[1,],p2[1,]), beside = TRUE, col = c("red", "blue")) 
#' @export
tidylda <- function(dtm, k, iterations = NULL, burnin = -1, alpha = 0.1, beta = 0.05, 
                    optimize_alpha = FALSE, calc_likelihood = FALSE, 
                    calc_r2 = FALSE, return_data = FALSE, ...) {
  
  UseMethod("tidylda")
  
}

#' @describeIn tidylda tidylda fit method for \code{dgCMatrix}
#' @export
tidylda.dgCMatrix <- function(dtm, ...) {
  
  # first, get the call for reproducibility
  mc <- match.call()
  
  ### check validity of inputs ----
  
  # iterations and burnin acceptable?
  if (burnin >= iterations) {
    stop("burnin must be less than iterations")
  }
  
  # dtm of the correct format?
  if (! "dgCMatrix" %in% class(dtm)) {
    message("dtm is not of class dgCMatrix, attempting to convert...")
    
    dtm <- try(methods::as(dtm, "dgCMatrix", strict = TRUE)) # requires Matrix in namespace
    
    if (! "dgCMatrix" %in% class(dtm))
      stop("conversion failed. Please pass an object of class dgCMatrix for dtm")
  }
  
  # is k formatted correctly?
  if (k < 2)
    stop("k must be 2 or greater")
  
  if (! is.numeric(k))
    stop("k must be an integer")
  
  k <- floor(k) # in case somebody is cheeky and passes a decimal
  
  # iterations?
  if (is.null(iterations))
    stop("You must specify number of iterations")
  
  # alpha and beta?
  alpha <- format_alpha(alpha = alpha, k = k)
  
  beta <- format_beta(beta = beta, k = k, Nv = ncol(dtm))
  
  if (! is.logical(calc_r2))
    stop("calc_r2 must be logical")
  
  ### format inputs ----
  
  
  # other formatting
  counts <- initialize_topic_counts(dtm = dtm, k = 10, 
                                    alpha = alpha$alpha, beta = beta$beta,
                                    ...)
  
  
  ### run C++ gibbs sampler ----
  lda <- fit_lda_c(docs = counts$docs,
                   Nk = k,
                   alpha = alpha$alpha,
                   beta = beta$beta,
                   Cd = counts$Cd,
                   Cv = counts$Cv,
                   Ck = counts$Ck,
                   Zd = counts$Zd,
                   Phi = counts$Cv, # this is actually ignored as freeze_topics = FALSE for initial fitting
                   iterations = iterations,
                   burnin = burnin,
                   freeze_topics = FALSE, # this stays FALSE for initial fitting 
                   calc_likelihood = calc_likelihood, 
                   optimize_alpha = optimize_alpha) 
  
  ### format the output ----
  
  result <- format_raw_lda_outputs(lda = lda, dtm = dtm, burnin = burnin, 
                                   is_prediction = FALSE, 
                                   alpha = alpha, beta = beta, 
                                   optimize_alpha = optimize_alpha, calc_r2 = calc_r2, 
                                   calc_likelihood = calc_likelihood, 
                                   call = mc, ...)
  
  ### return the result ----
  result
}