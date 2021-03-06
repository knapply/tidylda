% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{create_lexicon}
\alias{create_lexicon}
\title{Make a lexicon for looping over in the gibbs sampler}
\usage{
create_lexicon(Cd, Phi, dtm, alpha, freeze_topics, threads)
}
\arguments{
\item{Cd}{IntegerMatrix denoting counts of topics in documents}

\item{Phi}{NumericMatrix denoting probability of words in topics}

\item{dtm}{arma::sp_mat document term matrix}

\item{alpha}{NumericVector prior for topics over documents}

\item{freeze_topics}{bool if making predictions, set to \code{TRUE}}
}
\description{
One run of the Gibbs sampler and other magic to initialize some objects.
  Works in concert with \code{\link[tidylda]{initialize_topic_counts}}.
}
\keyword{internal}
