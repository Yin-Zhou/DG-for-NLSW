#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>


// prototype for LAPACK symmetric tridiagonal eigenvalue
extern void dsteqr_(
      char *COMPZ,     // 'T' to get eigenvectors too, 'N' otherwise
      int *N,          // order of matrix 
      double *D,       // diagonal elements of matrix
      double *E,       // sub-diagonal elements of matrix
      double *Z,       // matrix of eigenvectors, unused if COMPZ = 'N'
      int *LDZ,        // leading dim of Z
      double *WORK,    // working array, unused if COMPZ = 'N'
      int *INFO        // success flag
      );  


int lglnodes(int n, double *x, double *w)
/* calculates node locations and weights for n-point
   Gauss-Lobatto  quadrature.  It is assumed x and w are both size n. */
{ 
  double *e; // sub-diagonal of symmetric tridiagonal matrix
  char compz = 'N'; // job is to calculate eigenvalues only
  int ldz; // leading dimension of z for lapack
  int info; // return value, 0 = success
  int i1, i2; // loop counters
  if (n == 2 ) {
    x[0] = -1;
    x[1] = 1;
    w[0] = w[1] = 0.5;
    info = 0;
  }
  else if  ( n > 2 ) {
    ldz = n-2;
    e = malloc(n*sizeof(double));

    for( i1 = 0; i1 < ldz; i1++ ) 
      x[i1+1] = 0; // We use x for the diagonal elements
    for( i1 = 1; i1 < ldz; i1++ )
      e[i1-1] = sqrt(i1*(i1+2)/(2*i1+1.0)/(2*i1+3.0));

    dsteqr_(&compz, &ldz, &x[1], e, e, &ldz, e, &info);
    
    x[0] = -1;
    x[n-1] = 1;
    // calculate weights from known locations
    e[0] = 1;
    for( i1 = 0; i1 < n; i1++ ) {
      e[1] = x[i1];
      for( i2 = 2; i2 < n; i2++ )
        e[i2] = ((2*i2-1)*x[i1]*e[i2-1] - (i2-1)*e[i2-2])/i2;
      w[i1] = 2/(n*(n-1)*e[n-1]*e[n-1]);
    }
    free(e);
  }
  else
    info = -1;
    
  return(info);
}


int main() {
	int ord = 8;
}


