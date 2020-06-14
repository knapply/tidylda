#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::export]]
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


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.
//

/*** R
l <- list(
  matrix(0, nrow = 10, ncol = 3),
  matrix(1, nrow = 10, ncol = 2),
  matrix(2, nrow = 10, ncol = 5)
)

result <- cbind_list(l, 10)

result
*/
