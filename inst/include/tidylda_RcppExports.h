// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#ifndef RCPP_tidylda_RCPPEXPORTS_H_GEN_
#define RCPP_tidylda_RCPPEXPORTS_H_GEN_

#include <RcppArmadillo.h>
#include <Rcpp.h>

namespace tidylda {

    using namespace Rcpp;

    namespace {
        void validateSignature(const char* sig) {
            Rcpp::Function require = Rcpp::Environment::base_env()["require"];
            require("tidylda", Rcpp::Named("quietly") = true);
            typedef int(*Ptr_validate)(const char*);
            static Ptr_validate p_validate = (Ptr_validate)
                R_GetCCallable("tidylda", "_tidylda_RcppExport_validate");
            if (!p_validate(sig)) {
                throw Rcpp::function_not_exported(
                    "C++ function with signature '" + std::string(sig) + "' not found in tidylda");
            }
        }
    }

    inline Rcpp::List create_lexicon(arma::imat& Cd, const arma::mat& Phi, arma::sp_mat& dtm, const arma::vec alpha, const bool freeze_topics, const int threads) {
        typedef SEXP(*Ptr_create_lexicon)(SEXP,SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_create_lexicon p_create_lexicon = NULL;
        if (p_create_lexicon == NULL) {
            validateSignature("Rcpp::List(*create_lexicon)(arma::imat&,const arma::mat&,arma::sp_mat&,const arma::vec,const bool,const int)");
            p_create_lexicon = (Ptr_create_lexicon)R_GetCCallable("tidylda", "_tidylda_create_lexicon");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_create_lexicon(Shield<SEXP>(Rcpp::wrap(Cd)), Shield<SEXP>(Rcpp::wrap(Phi)), Shield<SEXP>(Rcpp::wrap(dtm)), Shield<SEXP>(Rcpp::wrap(alpha)), Shield<SEXP>(Rcpp::wrap(freeze_topics)), Shield<SEXP>(Rcpp::wrap(threads)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<Rcpp::List >(rcpp_result_gen);
    }

    inline Rcpp::List fit_lda_c(const std::vector<std::vector<int>>& docs, const int Nk, const arma::mat& beta, arma::vec& alpha, arma::umat& Cd, arma::mat& Cv, arma::uvec& Ck, std::vector<std::vector<int>>& Zd, const arma::mat& Phi, const int iterations, const int burnin, const bool freeze_topics, const bool calc_likelihood, const bool optimize_alpha) {
        typedef SEXP(*Ptr_fit_lda_c)(SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP,SEXP);
        static Ptr_fit_lda_c p_fit_lda_c = NULL;
        if (p_fit_lda_c == NULL) {
            validateSignature("Rcpp::List(*fit_lda_c)(const std::vector<std::vector<int>>&,const int,const arma::mat&,arma::vec&,arma::umat&,arma::mat&,arma::uvec&,std::vector<std::vector<int>>&,const arma::mat&,const int,const int,const bool,const bool,const bool)");
            p_fit_lda_c = (Ptr_fit_lda_c)R_GetCCallable("tidylda", "_tidylda_fit_lda_c");
        }
        RObject rcpp_result_gen;
        {
            RNGScope RCPP_rngScope_gen;
            rcpp_result_gen = p_fit_lda_c(Shield<SEXP>(Rcpp::wrap(docs)), Shield<SEXP>(Rcpp::wrap(Nk)), Shield<SEXP>(Rcpp::wrap(beta)), Shield<SEXP>(Rcpp::wrap(alpha)), Shield<SEXP>(Rcpp::wrap(Cd)), Shield<SEXP>(Rcpp::wrap(Cv)), Shield<SEXP>(Rcpp::wrap(Ck)), Shield<SEXP>(Rcpp::wrap(Zd)), Shield<SEXP>(Rcpp::wrap(Phi)), Shield<SEXP>(Rcpp::wrap(iterations)), Shield<SEXP>(Rcpp::wrap(burnin)), Shield<SEXP>(Rcpp::wrap(freeze_topics)), Shield<SEXP>(Rcpp::wrap(calc_likelihood)), Shield<SEXP>(Rcpp::wrap(optimize_alpha)));
        }
        if (rcpp_result_gen.inherits("interrupted-error"))
            throw Rcpp::internal::InterruptedException();
        if (Rcpp::internal::isLongjumpSentinel(rcpp_result_gen))
            throw Rcpp::LongjumpException(rcpp_result_gen);
        if (rcpp_result_gen.inherits("try-error"))
            throw Rcpp::exception(Rcpp::as<std::string>(rcpp_result_gen).c_str());
        return Rcpp::as<Rcpp::List >(rcpp_result_gen);
    }

}

#endif // RCPP_tidylda_RCPPEXPORTS_H_GEN_
