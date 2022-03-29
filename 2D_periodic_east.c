#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include "optimal_cosines.h"
#define IDXM(i, j) ((i) + (m+ord+1)*(j))
#define IDXQ(i, j) ((i) + (q+1)*(j))
#define IDXQPB(i, j, k, l) ((i) + (q+1)*(j) + (q+1)*(q+1)*(k) + (q+1)*(q+1)*(P+1)*(l))

// prototype for LAPACK general LU factorization
extern void dgetrf_(
      int *N,          // number of rows in matrix we want to factor 
      int *M,          // number of columns in matrix we want to factor 
      double *A,       // matrix we want to factor
      int *LDA,        // leading dim of A
      double *IPIV,    // size min(N,M) output vector of pivot indicies
      int *INFO        // success flag
      );  
// prototype for LAPACK general solve given LU factorization
extern void dgetrs_(
      char *TRANS,     // 'T' for transpose, 'N' otherwise
      int *N,          // order of matrix we want to divide by
      int *NRHS,       // number of right-hand sides 
      double *A,       // matrix we want to divide by
      int *LDA,        // leading dim of A
      double *IPIV,    // size N vector of pivot indicies
      double *B,       // right-hand side we want to solve for
      int *LDB,        // leading dim of B
      int *INFO        // success flag
      );  
// prototype for LAPACK banded triangular solve
extern void dtbtrs_(
      char *UPLO,      // upper or lower triangular?
      char *TRANS,     // 'T' for transpose, 'N' otherwise
      char *DIAG,      // 'U' for unit diagonal, 'N' otherwise
      int *N,          // order of matrix we want to solve for 
      int *KD,         // number of off diagonals 
      int *NRHS,       // number of right-hand sides 
      double *AB,      // bands of matrix we want to solve for
      int *LDAB,       // leading dim of AB
      double *B,       // right-hand side we want to solve for
      int *LDB,        // leading dim of B
      int *INFO        // success flag
      );  
// prototype for LAPACK symmetric eigenvalue
extern void dsyev_(
      char *JOBZ,      // calculate eigenvectors too?
      char *UPLO,      // upper or lower triangular?
      int *N,          // order of matrix 
      double *A,       // matrix whose eigenvalues we seek
      int *LDA,        // leading dim of A
      double *W,       // array for returning eigenvalues
      double *WORK,    // working array
      int *LWORK,      // length of working array  
      int *INFO        // success flag
      );  
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
// prototype for LAPACK Cholesky factorization
extern void dpotrf_(
      char *UPLO,      // upper or lower triangular?
      int *N,          // order of matrix 
      double *A,       // matrix to be factored 
      int *LDA,        // leading dim of A
      int *INFO        // success flag
      );  
// prototype for LAPACK spd solve with Cholesky factorization
extern void dpotrs_(
      char *UPLO,      // upper or lower triangular?
      int *N,          // order of matrix 
      int *NRHS,       // number of right-hand sides 
      double *A,       // pre-factored matrix 
      int *LDA,        // leading dim of A
      double *B,       // matrix of right-hand sides
      int *LDB,        // leading dim of B
      int *INFO        // success flag
      );

int lglnodes( int n, double *x, double *w )
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

int InterpEvenDeriv( int n, double h, double *M, double *pivot)
// Calculates an LU factored Hermite interpolation matrix
// for calculating a polynomial on the interval [-h, h] 
// from n even derivatives at each of x=-h and x=h.
// The outputs are intended for use with dgetrs_ and a vector D
// The first half of input vector D is the value at x=-h
// and even order derivatives at x = -h.
// The second half of input vector D is the value at x=h
// and even order derivatives at x = h.
// Then dgetrs_ will return the coefficients of a polynomial
// of degree 2n-1 in the 2n values of output vector D.
{
  int info; // return value, 0 = success
  int lda; // both equal to 2n
  int i1, i2, i3; // loop counters 

  lda = 2*n;  // M must be size (lda, lda) and pivot size (lda)

  for ( i1 = 0; i1 < n; i1++ )
    // left endpoint x = -h
    for ( i2 = 2*i1; i2 < 2*n; i2++ ) {
      M[i1 + lda*i2] = pow(-h, i2-2*i1); //each derivative reduces power of h
      for ( i3 = i2+1-2*i1; i3 <= i2; i3++ )
        M[i1 + lda*i2] *= i3; // each derivative increases coefficient
    }
  for ( i1 = 0; i1 < n; i1++ )
    // right endpoint x = h
    for ( i2 = 2*i1; i2 < 2*n; i2++ ) {
      M[i1+n + lda*i2] = pow(h, i2-2*i1); //each derivative reduces power of h
      for ( i3 = i2+1-2*i1; i3 <= i2; i3++ )
        M[i1+n + lda*i2] *= i3; // each derivative increases coefficient
    }

  dgetrf_(&lda, &lda, M, &lda, pivot, &info);
  if (info != 0)
    printf("Hermite interpolation failed with error code %d", info);
  return(info);        
}

int GaussQCofs( int n, double *x, double *w )
/* calculates node locations and weights for n-point Gaussian
   quadrature.  It is assumed x and w are both size n. */
{
  double *a; // upper triangular part of symmetric matrix
  double *work; // scratch space for llapack
  char job = 'V'; // job is to calculate eigenvectors too
  char uplo = 'U'; // upper triangular input
  int info; // return value, 0 = success
  int lda; // same as n 
  int lwork; // length of working array
  int i; // loop counter

  lda = n;
  a = calloc(n*n, sizeof(double));
  lwork = 3*n-1;
  work = malloc(lwork*sizeof(double));

  for( i=0; i<n-1; i++ ) 
    a[i+(i+1)*n] = (i+1)/sqrt(4.0*(i+1)*(i+1)-1.0);

  dsyev_(&job, &uplo, &lda, a, &lda, x, work, &lwork, &info);

  for( i=0; i<n; i++ )
    w[i] = 2*a[n*i]*a[n*i];

  free(a);
  free(work);
  return(info);
}
 
int LegPoly( int nodes, int q, double *x, double *y )
/* evaluate Legendre Polynomials of degree zero through q
   at the nodes in array x, returning values in array y.
   It is assumed that x is size nodes and that y is size
   nodes*(q+1).  */
{
  int i1, i2; // loop counters
  
  for( i1 = 0; i1 < nodes; i1++ )
    y[i1] = 1;
  if( q > 0 ) {
    for( i1 = 0; i1 < nodes; i1++ )
      y[i1+nodes] = x[i1];
    for( i2 = 2; i2 <= q; i2++ ) // three-term recurrence
      for( i1 = 0; i1 < nodes; i1++ )
        y[i1+nodes*i2] = ((2.0*i2-1.0)/i2)*x[i1]*y[i1+nodes*(i2-1)] \
          -((i2-1.0)/i2)*y[i1+nodes*(i2-2)];
  }
  return(0);
}

double sin_pi_x( double x, double complex amp )
{
  return(cimag(amp*cexp(I*M_PI*x)));
}

int LegendreLS( double(*funct)(double, double complex), double complex amp, \
  int q, double a, double b, double *coeff )
/* calculate the best fit in Legendre polynomials up to degree q
   for a given function within the interval [a b].
   It is assumed that coeff is size q+1.*/
{
  double *w; // weights of quadrature points
  double *x; // location of quadrature points
  double *y; // values of Legendre polynomials at quadrature points
  double *f; // values of approximated function at quadrature points
  int retval; // return value
  int i1, i2; // loop counters

  f = malloc((q+1)*sizeof(double));
  x = malloc((q+1)*sizeof(double));
  w = malloc((q+1)*sizeof(double));
  y = malloc((q+1)*(q+1)*sizeof(double));

  retval = GaussQCofs(q+1, x, w);
  retval = LegPoly(q+1, q, x, y);

  for( i1=0; i1<=q; i1++ )
    f[i1] = (*funct)((a + b + (b - a)*x[i1])/2.0, amp);

  for( i1=0; i1<=q; i1++ ) {
    coeff[i1] = 0;
    for( i2=0; i2<=q; i2++)
      coeff[i1] += w[i2]*f[i2]*y[i2+(q+1)*i1];
    coeff[i1] *= (i1 + 0.5);
  }

  free(x);
  free(y);
  free(w);
  free(f);
  return(retval);
}

int DAB_stage( double c, double bL, int q, int P, int boxes, \
  double *Ao, double *Ai, double *Bo, double *Bi, double *Do, double *Di, \
  double *u, double *v, double *u_t, double *v_t, \
  double *Mv, double *Su, double *Mu, double *Sv, int ldMu, \
  double *u_t_interface_lgl, double *u_x_interface_lgl, \
  double *basis_to_lgl, double *lgl_weight, bool is_periodic )
/* takes current values of u, and v as input,
   and values of u_t and u_x on one side as input,
   then returns derivatives in u_t and v_t */ 
{
  int info; // return value, 0 = success
  int nrhs; // number of right-hand sides for llapack
  int pp1; // order of matrix solve for llapack
  int ldab = 2; // leading dimension of AB for llapack
  int kd = 1; // number of off-diagonals for llapack
  char uplo; // upper-lower triangular flag for llapack
  char trans = 'N'; // transposition flag for llapack
  char diag = 'N'; // unit diagonal flag for llapack
  long i1; // big loop counter
  int i2, i3, i4, i5; // small loop counters
  double foo; // to hold intermediate multiplications
  double *AB; // left-hand side for calculating fluxes
  double *scratch; // to hold intermediate matrix multiplications
  double *scratch2; // to hold intermediate matrix multiplications
  // boundary fluxes: note indexing transposed from u and v
  double *V_star_o; // coefficients of basis functions
  double *V_star_i; // values at lgl nodes
  double *V_star_lr; // smaller because uncoupled, i.e. reusable
  double *n_dot_W_star_o; // coefficients of basis functions
  double *n_dot_W_star_i;  // values at lgl nodes
  double *n_dot_W_star_lr; // smaller because uncoupled, i.e. reusable

  pp1 = P+1;
  AB = malloc(2*(P+1)*sizeof(double));
  scratch = malloc((P+1)*(q+1)*sizeof(double));
  scratch2 = malloc((P+1)*(q+1)*sizeof(double));
  V_star_o = malloc((P+1)*(q+1)*sizeof(double));
  V_star_i = malloc((P+1)*(q+1)*sizeof(double));
  V_star_lr = malloc((q+1)*sizeof(double));
  n_dot_W_star_o = malloc((P+1)*(q+1)*sizeof(double));
  n_dot_W_star_i = malloc((P+1)*(q+1)*sizeof(double));
  n_dot_W_star_lr = malloc((q+1)*sizeof(double));

  // clear u_t and v_t
  for ( i1 = 0; i1 < (q+1)*(q+1)*(P+1)*boxes; i1++ ) {
    u_t[i1] = 0;
    v_t[i1] = 0;
  }

  for ( i5 = 0; i5 < boxes; i5++ ) { // for each box...
    /* OUTER FLUXES */
    // set up outside equation left-hand side (upper triangular)
    for ( i1 = 0; i1 < 2*(P+1); i1++ )
      AB[i1] = Ao[i1] + Bo[i1];
    uplo = 'U';
    
    // clear scratch
    for ( i1 = 0; i1 < (P+1)*(q+1); i1++ )
      scratch[i1] = 0;
    // set scratch to v evaluated on the outside 
    // minus c times normal derivative of u on the outside  
    for ( i3 = 0; i3 <= P; i3++ )
      for ( i2 = 0; i2 <= q; i2++ )  
        for ( i1 = 0; i1 <= q; i1++ )
          scratch[i3 + i2*(P+1)] += v[IDXQPB(i1, i2, i3, i5)] \
            - c*i1*(i1+1)*u[IDXQPB(i1, i2, i3, i5)]/bL;  

    // multiply scratch by Bo and place in V_star_o
    // multiply scratch by -Ao and place in n_dot_W_star_o
    for ( i2 = 0; i2 <= q; i2++ )  
      for ( i3 = 0; i3 <= P; i3++ ) {
        V_star_o[i3 + i2*(P+1)] = Bo[1+i3*2]*scratch[i3 + i2*(P+1)] \
          + (i3==P ? 0 : Bo[2*(i3+1)]*scratch[i3+1 + i2*(P+1)]); 
        n_dot_W_star_o[i3 + i2*(P+1)] = -Ao[1+i3*2]*scratch[i3 + i2*(P+1)] \
          + (i3==P ? 0 : -Ao[2*(i3+1)]*scratch[i3+1 + i2*(P+1)]); 
      }

    // clear scratch
    for ( i1 = 0; i1 < (P+1)*(q+1); i1++ )
      scratch[i1] = 0;
    // evaluate u on the outside and put in scratch
    for ( i3 = 0; i3 <= P; i3++ )
      for ( i2 = 0; i2 <= q; i2++ )  
        for ( i1 = 0; i1 <= q; i1++ )
          scratch[i3 + i2*(P+1)] += u[IDXQPB(i1, i2, i3, i5)];  

    // multiply scratch by Do and subtract from V_star_o and from n_dot_W_star_o
    for ( i2 = 0; i2 <= q; i2++ )  
      for ( i3 = 0; i3 <= P; i3++ ) {
        foo = Do[1+i3*2]*scratch[i3 + i2*(P+1)] + (i3==P ? 0 : Do[2*(i3+1)]*scratch[i3+1 + i2*(P+1)]);
        V_star_o[i3 + i2*(P+1)] -= foo;
        n_dot_W_star_o[i3 + i2*(P+1)] -= foo;
      }       

    nrhs = q+1;
    // divide V_star_o by (Ao + Bo)
    dtbtrs_( &uplo, &trans, &diag, &pp1, &kd, &nrhs, AB, &ldab, V_star_o, &pp1, &info);
    if (info != 0)
      printf("Divide V_star_o by (Ao + Bo) failed with error code %d", info);

    // divide n_dot_W_star_o by (Ao + Bo)
    dtbtrs_( &uplo, &trans, &diag, &pp1, &kd, &nrhs, AB, &ldab, n_dot_W_star_o, &pp1, &info);
    if (info != 0)
      printf("Divide n_dot_W_star_o by (Ao + Bo) failed with error code %d", info);
    
    /* INNER FLUXES */
    // set up inside equation left-hand side (lower triangular)
    for ( i1 = 0; i1 < 2*(P+1); i1++ )
      AB[i1] = Ai[i1] + Bi[i1];
    uplo = 'L';
    
    // clear scratch
    for ( i1 = 0; i1 < (P+1)*(q+1); i1++ )
      scratch[i1] = 0;
    // set scratch to the value of v on the inside
    // minus c times normal derivative of u on the inside
    for ( i3 = 0; i3 <= P; i3++ )
      for ( i2 = 0; i2 <= q; i2++ )  
        for ( i1 = 0; i1 <= q; i1++ )
          scratch[i3 + i2*(P+1)] += (i1%2==0 ? 1 : -1)*v[IDXQPB(i1, i2, i3, i5)] \
            - c*(i1%2==0 ? 1 : -1)*i1*(i1+1)*u[IDXQPB(i1, i2, i3, i5)]/bL;    

    // clear scratch2
    for ( i1 = 0; i1 < (P+1)*(q+1); i1++ )
      scratch2[i1] = 0;
    // multiply scratch by basis_to_lgl and place in scratch2
    for ( i1 = 0; i1 <= q; i1++ )
      for ( i2 = 0; i2 <= q; i2++ )  
        for ( i3 = 0; i3 <= P; i3++ )
          scratch2[i3 + i2*(P+1)] += scratch[i3 + i1*(P+1)]*basis_to_lgl[IDXQ(i2, i1)];  
    
    // multiply scratch2 by Bi and place in V_star_i
    // multiply scratch2 by -Ai and place in n_dot_W_star_i
    for ( i2 = 0; i2 <= q; i2++ )  
      for ( i3 = 0; i3 <= P; i3++ ) {
        V_star_i[i3 + i2*(P+1)] = (i3==0 ? 0 : Bi[-1+i3*2]*scratch2[i3-1 + i2*(P+1)]) \
          + Bi[2*i3]*scratch2[i3 + i2*(P+1)]; 
        n_dot_W_star_i[i3 + i2*(P+1)] = (i3==0 ? 0 : -Ai[-1+i3*2]*scratch2[i3-1 + i2*(P+1)]) \
          - Ai[2*i3]*scratch2[i3 + i2*(P+1)]; 
      }

    // clear scratch
    for ( i1 = 0; i1 < (P+1)*(q+1); i1++ )
      scratch[i1] = 0;
    // evaluate u on the inside and put in scratch
    for ( i3 = 0; i3 <= P; i3++ )
      for ( i2 = 0; i2 <= q; i2++ )  
        for ( i1 = 0; i1 <= q; i1++ )
          scratch[i3 + i2*(P+1)] += (i1%2==0 ? 1 : -1)*u[IDXQPB(i1, i2, i3, i5)];  

    // clear scratch2
    for ( i1 = 0; i1 < (P+1)*(q+1); i1++ )
      scratch2[i1] = 0;
    // multiply scratch by basis_to_lgl and place in scratch2
    for ( i1 = 0; i1 <= q; i1++ )
      for ( i2 = 0; i2 <= q; i2++ )  
        for ( i3 = 0; i3 <= P; i3++ )
          scratch2[i3 + i2*(P+1)] += scratch[i3 + i1*(P+1)]*basis_to_lgl[IDXQ(i2, i1)];  
    
    // multiply scratch2 by Di and subtract from V_star_i and from n_dot_W_star_i
    for ( i2 = 0; i2 <= q; i2++ )  
      for ( i3 = 0; i3 <= P; i3++ ) {
        foo = (i3==0 ? 0 : Di[-1+i3*2]*scratch2[i3-1 + i2*(P+1)]) + Di[2*i3]*scratch2[i3 + i2*(P+1)];
        V_star_i[i3 + i2*(P+1)] -= foo;
        n_dot_W_star_i[i3 + i2*(P+1)] -= foo;
      }

    // add (u_t - c*u_x), to the first component of V_star_i and n_dot_W_star_i
    for ( i2 = 0; i2 <= q; i2++ ) {
      foo = u_t_interface_lgl[i2 + i5*q] - c*u_x_interface_lgl[i2 + i5*q];
      V_star_i[i2*(P+1)] += foo;
      n_dot_W_star_i[i2*(P+1)] += foo;
    }
    // divide V_star_i by (Ai + Bi)
    dtbtrs_( &uplo, &trans, &diag, &pp1, &kd, &nrhs, AB, &ldab, V_star_i, &pp1, &info);
    if (info != 0)
      printf("Divide V_star_i by (Ai + Bi) failed with error code %d", info);

    // divide n_dot_W_star_i by (Ai + Bi)
    dtbtrs_( &uplo, &trans, &diag, &pp1, &kd, &nrhs, AB, &ldab, n_dot_W_star_i, &pp1, &info);
    if (info != 0)
      printf("Divide n_dot_W_star_i by (Ai + Bi) failed with error code %d", info);

    /* CALCULATE U_T */
    // put (kron(Sv, scale) + kron(scale, Sv))*v into u_t (negative of matlab rhs)
    for ( i4 = 0; i4 <= P; i4++ ) // auxiliary function number
      for ( i1 = 2; i1 <= q; i1++ )
        for ( i2 = i1-2; i2 >= 0; i2 -= 2 )
          for ( i3 = 0; i3 <= q; i3++ ) {
            u_t[IDXQPB(i1, i3, i4, i5)] += Sv[IDXQ(i1, i2)]*Mv[i3]*v[IDXQPB(i2, i3, i4, i5)];
            u_t[IDXQPB(i3, i1, i4, i5)] += Mv[i3]*Sv[IDXQ(i1, i2)]*v[IDXQPB(i3, i2, i4, i5)];
          }
    // special case equation (2.9)
    for ( i4 = 0; i4 <= P; i4++ )
      u_t[IDXQPB(0, 0, i4, i5)] = bL*bL*v[IDXQPB(0, 0, i4, i5)];
      
    // calculate and lift boundary flux

    for ( i4 = 0; i4 <= P; i4++ ) { // auxiliary function number
      for ( i1 = 0; i1 <= q; i1++ )
        V_star_lr[i1] = 0;
      if ( is_periodic || i5 != boxes-1 ) { // periodic or not leftmost box
        // average of velocity on left side
        for ( i2 = 0; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            V_star_lr[i1] += 0.5*(v[IDXQPB(i1, i2, i4, i5)] \
              + (i2%2==0 ? 1 : -1)*v[IDXQPB(i1, i2, i4, (i5+1)%boxes)]);
        // minus jump in normal derivative on left side
        for ( i2 = 1; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            V_star_lr[i1] -= (c*0.5)*(i2*(i2+1)*u[IDXQPB(i1, i2, i4, i5)]/bL \
              - (i2%2==0 ? -1 : 1)*i2*(i2+1)*u[IDXQPB(i1, i2, i4, (i5+1)%boxes)]/bL);
        // boundary integral on left side (2.7)
        for ( i2 = 1; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            u_t[IDXQPB(i1, i2, i4, i5)] += (i2*(i2+1)/bL)*Mv[i1]*V_star_lr[i1];
      }
    }
    
    for ( i4 = 0; i4 <= P; i4++ ) { // auxiliary function number
      for ( i1 = 0; i1 <= q; i1++ )
        V_star_lr[i1] = 0;
      if ( is_periodic || i5 != 0 ) { // periodic or not rightmost box
        // average velocity on right side
        for ( i2 = 0; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            V_star_lr[i1] += ((i2%2==0 ? 1 : -1)*v[IDXQPB(i1, i2, i4, i5)] \
              + v[IDXQPB(i1, i2, i4, (i5-1+boxes)%boxes)])/(2*c);
        // minus jump in normal derivative on right side
        for ( i2 = 1; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            V_star_lr[i1] -= (c*0.5)*((i2%2==0 ? 1 : -1)*i2*(i2+1)*u[IDXQPB(i1, i2, i4, i5)]/bL \
              + i2*(i2+1)*u[IDXQPB(i1, i2, i4, (i5-1+boxes)%boxes)]/bL);
        // boundary integral on right side (2.7)
        for ( i2 = 1; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            u_t[IDXQPB(i1, i2, i4, i5)] += \
              (i2%2==0 ? 1 : -1)*(i2*(i2+1)/bL)*Mv[i1]*V_star_lr[i1];
      }
    }

    // integrate outside flux (2.7)    
    for ( i4 = 0; i4 <= P; i4++ ) // auxiliary function number
      for ( i2 = 0; i2 <= q; i2++ )
        for ( i1 = 1; i1 <= q; i1++ )
          u_t[IDXQPB(i1, i2, i4, i5)] += (i1*(i1+1)/bL)*Mv[i2]*V_star_o[i4 + i2*(P+1)];
    
    // integrate inside flux (2.7) 
    // clear scratch
    for ( i1 = 0; i1 < (P+1)*(q+1); i1++ )
      scratch[i1] = 0;
    // quadrature along interface
    for ( i4 = 0; i4 <= P; i4++ ) // auxiliary function number
      for ( i2 = 0; i2 <= q; i2++ ) // tangential basis function index
        for ( i3 = 0; i3 <= q; i3++ ) // lgl node number
          scratch[i4 + i2*(P+1)] += lgl_weight[i3]*basis_to_lgl[IDXQ(i3,i2)] \
            *V_star_i[i4 + i3*(P+1)];
    // scale by normal derivative
    for ( i4 = 0; i4 <= P; i4++ ) // auxiliary function number
      for ( i2 = 0; i2 <= q; i2++ ) // tangential basis function index
        for ( i1 = 1; i1 <= q; i1++ ) // normal basis function index
          u_t[IDXQPB(i1, i2, i4, i5)] += (i1%2==0 ? 1 : -1)*(i1*(i1+1)/bL)*scratch[i4 + i2*(P+1)];
      
    // call llapack matrix solve to divide by Mu
    nrhs = P+1;
    uplo = 'U';
    dpotrs_( &uplo, &ldMu, &nrhs, Mu, &ldMu, &u_t[i5*(q+1)*(q+1)*(P+1)], &ldMu, &info );
    if (info != 0)
      printf("Divide u_t by Mu failed with error code %d", info);

    /* CALCULATE V_T */
    // put -(kron(Su, scale) + kron(scale, Su))*u into v_t
    for ( i4 = 0; i4 <= P; i4++ ) // auxiliary function number
      for ( i1 = 1; i1 <= q; i1++ )
        for ( i2 = (i1-1)%2+1; i2 <= q; i2 += 2 ) // hit only trimmed checkerboard
          for ( i3 = 0; i3 <= q; i3++ ) {
            v_t[IDXQPB(i1, i3, i4, i5)] -= \
              Su[IDXQ(i1, i2)]*Mv[i3]*u[IDXQPB(i2, i3, i4, i5)];
            v_t[IDXQPB(i3, i1, i4, i5)] -= \
              Mv[i3]*Su[IDXQ(i1, i2)]*u[IDXQPB(i3, i2, i4, i5)];
          }

    // calculate and lift boundary flux

    for ( i4 = 0; i4 <= P; i4++ ) { // auxiliary function number
      // average of normal derivative on left side
      for ( i1 = 0; i1 <= q; i1++ )
        n_dot_W_star_lr[i1] = 0;
      if ( i5 == boxes-1 && !is_periodic ) // leftmost box and not periodic
        for ( i2 = 1; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            n_dot_W_star_lr[i1] += i2*(i2+1)*u[IDXQPB(i1, i2, i4, i5)]/bL;
      else // periodic or not leftmost box
        for ( i2 = 1; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            n_dot_W_star_lr[i1] += 0.5*(i2*(i2+1)*u[IDXQPB(i1, i2, i4, i5)]/bL \
              + (i2%2==0 ? -1 : 1)*i2*(i2+1)*u[IDXQPB(i1, i2, i4, (i5+1)%boxes)]/bL);
      // minus jump in velocity on left side
      if ( is_periodic || i5 != boxes-1 ) // periodic or not leftmost box
        for ( i2 = 0; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            n_dot_W_star_lr[i1] -= (v[IDXQPB(i1, i2, i4, i5)] \
              - (i2%2==0 ? 1 : -1)*v[IDXQPB(i1, i2, i4, (i5+1)%boxes)])/(2*c);
      // boundary integral on left side (2.8)
      for ( i2 = 0; i2 <= q; i2++ )
        for ( i1 = 0; i1 <= q; i1++ )
          v_t[IDXQPB(i1, i2, i4, i5)] += c*c*Mv[i1]*n_dot_W_star_lr[i1];
    }
    
    for ( i4 = 0; i4 <= P; i4++ ) { // auxiliary function number
      // average of normal derivative on right side
      for ( i1 = 0; i1 <= q; i1++ )
        n_dot_W_star_lr[i1] = 0;
      if ( i5 == 0 && !is_periodic ) // rightmost box and not periodic
        for ( i2 = 1; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            n_dot_W_star_lr[i1] += \
              (i2%2==0 ? 1 : -1)*i2*(i2+1)*u[IDXQPB(i1, i2, i4, i5)]/bL;
      else // periodic or not rightmost box
        for ( i2 = 1; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            n_dot_W_star_lr[i1] += 0.5*(-i2*(i2+1)*u[IDXQPB(i1, i2, i4, (i5-1+boxes)%boxes)]/bL \
              + (i2%2==0 ? 1 : -1)*i2*(i2+1)*u[IDXQPB(i1, i2, i4, i5)]/bL);
      // minus jump in velocity on right side
      if ( is_periodic || i5 != 0 ) // periodic or not rightmost box
        for ( i2 = 0; i2 <= q; i2++ )
          for ( i1 = 0; i1 <= q; i1++ )
            n_dot_W_star_lr[i1] -= ((i2%2==0 ? 1 : -1)*v[IDXQPB(i1, i2, i4, i5)] \
              - v[IDXQPB(i1, i2, i4, (i5-1+boxes)%boxes)])/(2*c);
      // boundary integral on right side (2.8)
      for ( i2 = 0; i2 <= q; i2++ )
        for ( i1 = 0; i1 <= q; i1++ )
          v_t[IDXQPB(i1, i2, i4, i5)] += \
            (i2%2==0 ? 1 : -1)*c*c*Mv[i1]*n_dot_W_star_lr[i1];
    }
      
    // integrate outside flux (2.8)    
    for ( i4 = 0; i4 <= P; i4++ ) // auxiliary function number 
      for ( i2 = 0; i2 <= q; i2++ )
        for ( i1 = 0; i1 <= q; i1++ )
          v_t[IDXQPB(i1, i2, i4, i5)] += \
            c*c*Mv[i2]*n_dot_W_star_o[i4 + i2*(P+1)];
    
    // integrate inside flux (2.8) 
    // clear scratch
    for ( i1 = 0; i1 < (P+1)*(q+1); i1++ )
      scratch[i1] = 0;
    // quadrature along interface
    for ( i4 = 0; i4 <= P; i4++ ) // auxiliary function number
      for ( i2 = 0; i2 <= q; i2++ ) // tangential basis function index
        for ( i3 = 0; i3 <= q; i3++ ) // lgl node number
          scratch[i4 + i2*(P+1)] += lgl_weight[i3]*basis_to_lgl[IDXQ(i3,i2)] \
            *n_dot_W_star_i[i4 + i3*(P+1)];
    // scale by normal derivative
    for ( i4 = 0; i4 <= P; i4++ ) // auxiliary function number
      for ( i2 = 0; i2 <= q; i2++ ) // tangential basis function index
        for ( i1 = 0; i1 <= q; i1++ ) // normal basis function index
          v_t[IDXQPB(i1, i2, i4, i5)] += (i1%2==0 ? 1 : -1)*c*c*scratch[i4 + i2*(P+1)];

    // divide by kron(Mv, Mv)
    for ( i4 = 0; i4 <= P; i4++ ) // auxiliary function number
      for ( i2 = 0; i2 <= q; i2++ )
        for ( i1 = 0; i1 <= q; i1++ )
          v_t[IDXQPB(i1, i2, i4, i5)] /= Mv[i1]*Mv[i2];
  } // end boxes

  free(AB);
  free(scratch);
  free(scratch2);
  free(V_star_o);
  free(V_star_i);
  free(V_star_lr);
  free(n_dot_W_star_o);
  free(n_dot_W_star_i);
  free(n_dot_W_star_lr);

  return(info);
}

int main()
/* This program evolves the wave equation in second-order form
in a box with Dirichlet boundaries left, top, and bottom
and with a DAB on the right (i.e. positive x-direction).
The interior is evolved with a modified equation discretization.
There is a larger domain to serve as a reference solution.
The DAB uses the Appelo/Hagstrom DG discretization
with tensor products of Legendre polynomials as basis functions
and classic RK4 for timestepping.
It's a DAB with P auxiliary functions coupled at x=L and x=L+bL.
The interior solution is continued in zeroth auxiliary function
with information u_x and u_t at x=L passed to DAB in the form of
a polynomial in time to accommodate smaller timesteps of DAB.
Sommerfeld termination of Pth auxiliary function at x=L+bL.
Initial condition is a stationary pulse near the center of
the interior. */
{
  int ord = 8; // order of accuracy; must be 2, 4, 6, or 8
  int ng; // number of ghost points = ord/2;
  ng = ord/2;
  double c = 1; // wave speed
  int q; // highest degree polynomial in Legendre basis
  q = (ord == 2) ? 3 : ord-1; // we need at least q=3 even for order 2
  int P = 7; // number of aux solutions in addition to interior continuation
  long factorial[11] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800};
  double a_d; // strength of artificial dissipation in interior
  a_d = 2.0*factorial[ng]*factorial[ng]/factorial[ord+2];
  double Tf = 1; // final time
  double L = 1; // length and width of interior
  int refine_lo = 64; // coarsest grid
  int refine_hi = 64; // finest grid
  int refine_step = 12; // grid refinement step
  bool periodic_ns = true; // whether north and south boundaries are periodic
  double L_ref; // width of reference domain (same height as interior)
  int h_per_cell; // ratio of DAB cell width to interior cell width
  h_per_cell = ng; //seems most efficient 
  int npilgl; // number of points interpolated to lgl nodes = 1+ng*2+h_per_cell
  npilgl = 1 + ng*2 + h_per_cell;
  int m; // number of intervals in each dimension of interior
  int m_ref; // number of intervals in width of reference interior
  double h; // length of one interval in interior = L/m
  double bL; // length and width of one DAB cell = h_per_cell*h
  // code breaks unless m is a multiple of h_per_cell
  int boxes; // number of coupled cells in DAB = m/h_per_cell
  double k; // size of one interior time step
  double ks; // size of one DAB time step
  int substep, substeps; // number of DAB time steps per interior time step = k/ks
  double t; // current time
  double t_now; // current time at RK stage
  double xloc, yloc; // variables to store coordinates of a grid point
  double old_err, err, foo; // errors for calculating convergence order
  int max_err_loc[2]; // coordinates of maximum error
  double err_bound; // error bound for optimal cosines
  double *u_int; // values on the grid points and 3 ghost points each side
  double *up_int; // values on the grid points at future time
  double *um_int; // values on the grid points at past time
  double *u_ref; // reference interior solution
  double *up_ref; // reference interior solution at future time
  double *um_ref; // reference interior solution at past time
  double *temp; // variable to recycle (rather than copy) arrays
  double *u; // basis function coefficients for displacement 
  double *v; // basis function coefficients for velocity
  double *u_t; // basis function coefficients for displacement 
  double *v_t; // basis function coefficients for velocity
  double *u_del; // basis function coefficients for displacement 
  double *v_del; // basis function coefficients for velocity
  double *u_now; // basis function coefficients for displacement 
  double *v_now; // basis function coefficients for velocity
  // even time derivatives of um and u at right edge grid points 
  double *u_deriv_right_grid;
  // even time derivatives of um and u at right edge lgl nodes 
  double *u_deriv_right_lgl;
  // even time derivatives of um_x and u_x at right edge grid points 
  double *u_x_deriv_right_grid;
  // even time derivatives of um_x and u_x at right edge lgl nodes 
  double *u_x_deriv_right_lgl;
  double *Mu; // (2D) mass matrix for u_t, equation (5.4) 
  double *Mv; // (diagonal) mass matrix for v_t, equation (5.3)
  double *Su; // stiffness matrix for u, equation (5.3)
  double *Sv; // stiffness matrix for v, equation (5.4)
  double *Ao; // A*U_t + cB*U_x + D*U = 0 exterior boundary condition 
  double *Ai; // A*U_t - cB*U_x + D*U = 0 interior boundary condition
  double *Bo; // A*U_t + cB*U_x + D*U = 0 exterior boundary condition
  double *Bi; // A*U_t - cB*U_x + D*U = 0 interior boundary condition
  double *Do; // A*U_t + cB*U_x + D*U = 0 exterior boundary condition
  double *Di; // A*U_t - cB*U_x + D*U = 0 interior boundary condition
  double *lgl_loc; // location of Gauss-Lobatto quadrature nodes
  double *lgl_weight; // weight of Gauss-Lobatto quadrature nodes
  double *bw; // barycentric weights used to calculate grid_to_lgl
  double *grid_to_lgl; // matrix to interpolate npilgl grid points into lgl nodes
  double *basis_to_lgl; // matrix to evaluate a vector of basis coeffs at lgl nodes
  double *ghost_loc; // scaled location of ghost points
  double *basis_to_ghost; // evaluation of Legendre polynomials as ghost nodes
  double *Herm_u; // matrix for interpolating even time derivatives of u
  double *Herm_u_pivot; // Pivot vector for factored Herm_u
  int ldHerm_u; // size of Herm_u for lapack call
  ldHerm_u = ord + 2;
  double *Herm_u_x; // matrix for interpolating even time derivatives of u_x
  double *Herm_u_x_pivot; // pivot vector for factored Herm_u_x
  int ldHerm_u_x; // size of Herm_u_x for lapack call
  ldHerm_u_x = ord;
  double *u_t_interface_lgl; // value of u_t at lgl nodes passed to DAB  
  double *u_x_interface_lgl; // value of u_x at lgl nodes passed to DAB  
  double rktimes[4] = {0, 1.0/2.0, 1.0/2.0, 1}; // diagonal Butcher tableau
  double rkweights[4] = {1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0};
  long i1; // big loop counter
  int i2, i3, i4, i5, i6; // small loop counters
  int stage; // RK stage counter
  char trans = 'N'; // flag for no transposition for lapack call
  char uplo = 'U'; // upper triangular input for lapack call
  int nrhs; // number of right-hand sides for llapack
  int info; // return value from lapack, 0 = success
  int ldMu; // leading dimension of Mu for lapack call
  int retval; // holder for function return value
  double deriv_factor; // save time by not recomputing factorial in innermost loop
  // binomial coefficient a.k.a. "n choose k" a.k.a. Pascal's Triangle
  int binom[5][5] = { {1, 0, 0, 0, 0}, {1, 1, 0, 0, 0}, {1, 2, 1, 0, 0}, \
    {1, 3, 3, 1, 0}, {1, 4, 6, 4, 1} }; 
  double cfdc[9][4][9]; // central finite difference coefficients
  /* The first index is the order of the derivative, where 0 is the
   * zeroth derivative, i.e. the value.  The second index represents
   * the order accuracy, which we compress due to having only accuracies
   * of even order in central finite differences.  Thus index 0 is
   * second-order accuracy, index 1 is fourth-order accuracy, and
   * index 2 is sixth-order accuracy.
   * The third index indicates which point in a nine-point stencil.
   */
  cfdc[0][0][4] = 1;
  cfdc[0][1][3] = 0;
  cfdc[0][1][4] = 1;
  cfdc[0][1][5] = 0;
  cfdc[0][2][2] = 0;
  cfdc[0][2][3] = 0;
  cfdc[0][2][4] = 1;
  cfdc[0][2][5] = 0;
  cfdc[0][2][6] = 0;
  cfdc[0][3][1] = 0;
  cfdc[0][3][2] = 0;
  cfdc[0][3][3] = 0;
  cfdc[0][3][4] = 1;
  cfdc[0][3][5] = 0;
  cfdc[0][3][6] = 0;
  cfdc[0][3][7] = 0;
  cfdc[1][0][3] = -1.0/2.0;
  cfdc[1][0][4] = 0;
  cfdc[1][0][5] = 1.0/2.0;
  cfdc[1][1][2] = 1.0/12.0;
  cfdc[1][1][3] = -2.0/3.0;
  cfdc[1][1][4] = 0;
  cfdc[1][1][5] = 2.0/3.0;
  cfdc[1][1][6] = -1.0/12.0;
  cfdc[1][2][1] = -1.0/60.0;
  cfdc[1][2][2] = 3.0/20.0;
  cfdc[1][2][3] = -3.0/4.0;
  cfdc[1][2][4] = 0;
  cfdc[1][2][5] = 3.0/4.0;
  cfdc[1][2][6] = -3.0/20.0;
  cfdc[1][2][7] = 1.0/60.0;
  cfdc[1][3][0] = 1.0/280.0;
  cfdc[1][3][1] = -4.0/105.0;
  cfdc[1][3][2] = 1.0/5.0;
  cfdc[1][3][3] = -4.0/5.0;
  cfdc[1][3][4] = 0;
  cfdc[1][3][5] = 4.0/5.0;
  cfdc[1][3][6] = -1.0/5.0;
  cfdc[1][3][7] = 4.0/105.0;
  cfdc[1][3][8] = -1.0/280.0;
  cfdc[2][0][3] = 1;
  cfdc[2][0][4] = -2;
  cfdc[2][0][5] = 1;
  cfdc[2][1][2] = -1.0/12.0;
  cfdc[2][1][3] = 4.0/3.0;
  cfdc[2][1][4] = -5.0/2.0;
  cfdc[2][1][5] = 4.0/3.0;
  cfdc[2][1][6] = -1.0/12.0;
  cfdc[2][2][1] = 1.0/90.0;
  cfdc[2][2][2] = -3.0/20.0;
  cfdc[2][2][3] = 3.0/2.0;
  cfdc[2][2][4] = -49.0/18.0;
  cfdc[2][2][5] = 3.0/2.0;
  cfdc[2][2][6] = -3.0/20.0;
  cfdc[2][2][7] = 1.0/90.0;
  cfdc[2][3][0] = -1.0/560.0;
  cfdc[2][3][1] = 8.0/315.0;
  cfdc[2][3][2] = -1.0/5.0;
  cfdc[2][3][3] = 8.0/5.0;
  cfdc[2][3][4] = -205.0/72.0;
  cfdc[2][3][5] = 8.0/5.0;
  cfdc[2][3][6] = -1.0/5.0;
  cfdc[2][3][7] = 8.0/315.0;
  cfdc[2][3][8] = -1.0/560.0;
  cfdc[3][0][2] = -1.0/2.0;
  cfdc[3][0][3] = 1;
  cfdc[3][0][4] = 0;
  cfdc[3][0][5] = -1;
  cfdc[3][0][6] = 1.0/2.0;
  cfdc[3][1][1] = 1.0/8.0;
  cfdc[3][1][2] = -1;
  cfdc[3][1][3] = 13.0/8.0;
  cfdc[3][1][4] = 0;
  cfdc[3][1][5] = -13.0/8.0;
  cfdc[3][1][6] = 1;
  cfdc[3][1][7] = -1.0/8.0;
  cfdc[3][2][0] = -7.0/240.0;
  cfdc[3][2][1] = 3.0/10.0;
  cfdc[3][2][2] = -169.0/120.0;
  cfdc[3][2][3] = 61.0/30.0;
  cfdc[3][2][4] = 0;
  cfdc[3][2][5] = -61.0/30.0;
  cfdc[3][2][6] = 169.0/120.0;
  cfdc[3][2][7] = -3.0/10.0;
  cfdc[3][2][8] = 7.0/240.0;
  cfdc[4][0][2] = 1;
  cfdc[4][0][3] = -4;
  cfdc[4][0][4] = 6;
  cfdc[4][0][5] = -4;
  cfdc[4][0][6] = 1;
  cfdc[4][1][1] = -1.0/6.0;
  cfdc[4][1][2] = 2;
  cfdc[4][1][3] = -13.0/2.0;
  cfdc[4][1][4] = 28.0/3.0;
  cfdc[4][1][5] = -13.0/2.0;
  cfdc[4][1][6] = 2;
  cfdc[4][1][7] = -1.0/6.0;
  cfdc[4][2][0] = 7.0/240.0;
  cfdc[4][2][1] = -2.0/5.0;
  cfdc[4][2][2] = 169.0/60.0;
  cfdc[4][2][3] = -122.0/15.0;
  cfdc[4][2][4] = 91.0/8.0;
  cfdc[4][2][5] = -122.0/15.0;
  cfdc[4][2][6] = 169.0/60.0;
  cfdc[4][2][7] = -2.0/5.0;
  cfdc[4][2][8] = 7.0/240.0;
  cfdc[5][0][1] = -1.0/2.0;
  cfdc[5][0][2] = 2;
  cfdc[5][0][3] = -5.0/2.0;
  cfdc[5][0][4] = 0;
  cfdc[5][0][5] = 5.0/2.0;
  cfdc[5][0][6] = -2;
  cfdc[5][0][7] = 1.0/2.0;
  cfdc[5][1][0] = 1.0/6.0;
  cfdc[5][1][1] = -3.0/2.0;
  cfdc[5][1][2] = 13.0/3.0;
  cfdc[5][1][3] = -29.0/6.0;
  cfdc[5][1][4] = 0;
  cfdc[5][1][5] = 29.0/6.0;
  cfdc[5][1][6] = -13.0/3.0;
  cfdc[5][1][7] = 3.0/2.0;
  cfdc[5][1][8] = -1.0/6.0;
  cfdc[6][0][1] = 1;
  cfdc[6][0][2] = -6;
  cfdc[6][0][3] = 15;
  cfdc[6][0][4] = -20;
  cfdc[6][0][5] = 15;
  cfdc[6][0][6] = -6;
  cfdc[6][0][7] = 1;
  cfdc[6][1][0] = -1.0/4.0;
  cfdc[6][1][1] = 3;
  cfdc[6][1][2] = -13;
  cfdc[6][1][3] = 29;
  cfdc[6][1][4] = -75.0/2.0;
  cfdc[6][1][5] = 29;
  cfdc[6][1][6] = -13;
  cfdc[6][1][7] = 3;
  cfdc[6][1][8] = -1.0/4.0;
  cfdc[7][0][0] = -1.0/2.0;
  cfdc[7][0][1] = 3;
  cfdc[7][0][2] = -7;
  cfdc[7][0][3] = 7;
  cfdc[7][0][4] = 0;
  cfdc[7][0][5] = -7;
  cfdc[7][0][6] = 7;
  cfdc[7][0][7] = -3;
  cfdc[7][0][8] = 1.0/2.0;
  cfdc[8][0][0] = 1;
  cfdc[8][0][1] = -8;
  cfdc[8][0][2] = 28;
  cfdc[8][0][3] = -56;
  cfdc[8][0][4] = 70;
  cfdc[8][0][5] = -56;
  cfdc[8][0][6] = 28;
  cfdc[8][0][7] = -8;
  cfdc[8][0][8] = 1;

//for ( P = 3; P < 8; P++ ) {
//for ( q = 3; q < 8; q++ ) {

  /* set up boundary condition matrices */
  Ao = malloc(2*(P+1)*sizeof(double)); // Store just two diagonals
  Ai = malloc(2*(P+1)*sizeof(double)); // Store just two diagonals
  Bo = malloc(2*(P+1)*sizeof(double)); // Store just two diagonals
  Bi = malloc(2*(P+1)*sizeof(double)); // Store just two diagonals
  Do = malloc(2*(P+1)*sizeof(double)); // Store just two diagonals
  Di = malloc(2*(P+1)*sizeof(double)); // Store just two diagonals

  retval = optimal_cosinesP(0.05, P, Bo, &err_bound);
//  retval = optimal_cosinesP(L/Tf, P, Bo, &err_bound);
  if (retval != 0)
    printf("Optimal cosines failure: flag = %i \n", retval);
  // We temporarily use Bo and Bi to store alpha and sigma respectively
  for ( i1 = 0; i1 < 2*P; i1++ )
    Bi[i1] = (1 - Bo[i1]*Bo[i1])/(Bo[i1]*Tf);
  // Lapack convention has matrix diagonals stored as rows
  for ( i1 = 0; i1 < P; i1++ ) { 
    Ao[1+2*i1] = Bo[2*i1];
    Ao[2*(i1+1)] = -Bo[2*i1+1];
    Do[1+2*i1] = Bi[2*i1];
    Do[2*(i1+1)] = -Bi[2*i1+1];
    Ai[1+2*i1] = -Bo[2*i1];
    Ai[2*(i1+1)] = Bo[2*i1+1];
    Di[1+2*i1] = -Bi[2*i1];
    Di[2*(i1+1)] = Bi[2*i1+1];
  }
  Ao[0] = 0; // super-diagonal is one shorter
  Ao[1+2*P] = 1; // Sommerfeld termination
  Do[0] = 0; // super-diagonal is one shorter
  Do[1+2*P] = 0; // Sommerfeld termination
  Ai[0] = 1; // get data from volume
  Ai[1+2*P] = 0; // sub-diagonal is one shorter
  Di[0] = 0; // get data from volume
  Di[1+2*P] = 0; // sub-diagonal is one shorter

  // Now put the correct values in Bo and Bi
  for ( i1 = 0; i1 < 2*(P+1); i1++ ) { 
    Bo[i1] = 1;
    Bi[i1] = 1;
  }
  Bo[0] = 0; // super-diagonal is one shorter
  Bi[1+2*P] = 0; // sub-diagonal is one shorter
  
  old_err = 0; // store previous error to calculate convergence order
  for ( m = refine_lo; m <= refine_hi; m += refine_step ) {
  // test different grid refinements
    h = L/m; // length of one interval
    bL = h_per_cell*h; // side length of one DAB cell
    boxes = m/h_per_cell; // number of cells in DAB
    k = 0.5*h/c; // size of one interior time step
    ks = bL/((q+1)*(q+1)*c); // maximum size of one DAB time step
    substeps = ceil(k/ks); // minimum number of substeps
    ks = k/substeps; // actual size of one DAB time step
    L_ref = L + Tf/(2*c); // wide enough that reflections don't return
    m_ref = ceil(L_ref/h); // minimum number of intervals
    L_ref = h*m_ref; // make reference length a multiple of h

    // note that the y-coordinate of the interior variables comes first
    u_int = malloc((m+ord+1)*(m+ord+1)*sizeof(double));
    up_int = malloc((m+ord+1)*(m+ord+1)*sizeof(double));
    um_int = malloc((m+ord+1)*(m+ord+1)*sizeof(double));
    u_ref = malloc((m+ord+1)*(m_ref+ord+1)*sizeof(double));
    up_ref = malloc((m+ord+1)*(m_ref+ord+1)*sizeof(double));
    um_ref = malloc((m+ord+1)*(m_ref+ord+1)*sizeof(double));

    // initialize interior with slightly off-center Gaussian pulse
    // that is stationary (i.e. same in u and um) 
    // and essentially zero by the boundaries
    for ( i2 = 0; i2 < m+ord+1; i2++ ) {
      xloc = h*(i2 - ng);
      for ( i1 = 0; i1 < m+ord+1; i1++ ) {
        yloc = h*(i1 - ng);
        um_ref[IDXM(i1, i2)] = u_ref[IDXM(i1, i2)] = um_int[IDXM(i1, i2)] = \
          u_int[IDXM(i1, i2)] = exp(-170*((xloc-0.51)*(xloc-0.51)+(yloc-0.52)*(yloc-0.52)));
      }
    }
    // rest of reference solution is essentially zero anyway
    for ( i2 = m+ord+1; i2 < m_ref+ord+1; i2++ )
      for ( i1 = 0; i1 < m+ord+1; i1++ )
        um_ref[IDXM(i1, i2)] = u_ref[IDXM(i1, i2)] = 0;

    // set up spatial iterpolation from grid points to Gauss-Lobatto nodes
    // We will use npilgl grid points: h_per_cell+1 from the box and ng to each side.
    // We use just enough (I hope!) lgl nodes to exactly integrate our
    // boundary polynomials on each box of the DAB.  That integration
    // also requires us to evaluate DAB vectors at lgl nodes via matrix basis_to_lgl  
    lgl_loc = malloc((q+1)*sizeof(double));
    lgl_weight = malloc((q+1)*sizeof(double));
    grid_to_lgl = malloc((q-1)*npilgl*sizeof(double));
    basis_to_lgl = malloc((q+1)*(q+1)*sizeof(double));
    ghost_loc = malloc((h_per_cell+1)*sizeof(double));
    basis_to_ghost = malloc((h_per_cell+1)*(q+1)*sizeof(double));
    
    for ( i1 = 0; i1 <= h_per_cell; i1++ )
      ghost_loc[i1] = -1 + i1*2.0/h_per_cell; // evenly spaced across width 2
      
    // get matrix for evaluating vector of DAB coefficients at ghost nodes
    retval = LegPoly( h_per_cell+1, q, ghost_loc, basis_to_ghost );

    retval = lglnodes(q+1, lgl_loc, lgl_weight);
    if (retval != 0)
      printf("Gauss-Lobatto nodes failure: flag = %i \n", retval);

    // get matrix for evaluating vector of DAB coefficients at lgl nodes
    retval = LegPoly( q+1, q, lgl_loc, basis_to_lgl );

    for ( i1 = 0; i1 <= q; i1++ ) { // scale to width of box
      lgl_loc[i1] *= bL/2.0;
      lgl_weight[i1] *= bL/2.0;
    }

    // set up Lagrange interpolation matrix via barycentric weights
    bw = malloc(npilgl*sizeof(double));
    for ( i1 = 0; i1 < npilgl; i1++ )
      bw[i1] = 1;
    for ( i1 = 0; i1 < npilgl; i1++ )
      for ( i2 = 0; i2 < npilgl; i2++ )
        if ( i1 != i2 )
          bw[i1] /= h*(i1 - i2);
    for ( i1 = 1; i1 < q; i1++ ) { // barycentric formula doesn't work at nodes
      foo = 0; // row sum
      for ( i2 = 0; i2 < npilgl; i2++ )
        foo += grid_to_lgl[i1-1 + (q-1)*i2] = bw[i2]/(lgl_loc[i1] - h*(i2-(npilgl-1)/2.0));
      for ( i2 = 0; i2 < npilgl; i2++ ) // divide row by row sum
        grid_to_lgl[i1-1 + (q-1)*i2] /= foo;
    }

    // get matrices for hermite interpolation in time
    Herm_u = calloc((ord+2)*(ord+2), sizeof(double));
    Herm_u_pivot = calloc(ord+2, sizeof(double));
    Herm_u_x = calloc(ord*ord, sizeof(double));
    Herm_u_x_pivot = calloc(ord, sizeof(double));
    
    retval = InterpEvenDeriv(ng+1, k/2, Herm_u, Herm_u_pivot);
    retval = InterpEvenDeriv(ng, k/2, Herm_u_x, Herm_u_x_pivot);

    // allocate variables for passing data from interior to DAB
    // let them be zero at time t=0 
    u_deriv_right_grid = calloc((ord+2)*(m+ord+1), sizeof(double));
    u_x_deriv_right_grid = calloc(ord*(m+ord+1), sizeof(double));
    u_deriv_right_lgl = malloc((ord+2)*(boxes*q+1)*sizeof(double));
    u_x_deriv_right_lgl = malloc(ord*(boxes*q+1)*sizeof(double));
    u_t_interface_lgl = malloc((boxes*q+1)*sizeof(double));
    u_x_interface_lgl = malloc((boxes*q+1)*sizeof(double));

    /* Calculate mass and stiffness matrices for DAB*/
    Mu = calloc((q+1)*(q+1)*(q+1)*(q+1), sizeof(double)); // Store 2D Cholesky factorization
    Mv = malloc((q+1)*sizeof(double)); // Store just the diagonal
    Su = calloc((q+1)*(q+1), sizeof(double));
    Sv = calloc((q+1)*(q+1), sizeof(double));

    // note that Mv is diagonal
    for ( i1 = 0; i1 <= q; i1++ )
      Mv[i1] = bL/(2.0*i1 + 1.0);

    for ( i1 = 1; i1 <= q; i1++ ) {
      Su[IDXQ(i1, i1)] = 2.0*c*c*i1*(i1+1)/bL;
      for ( i2 = i1+2; i2 <= q; i2 += 2) {
         Su[IDXQ(i1, i2)] = Su[IDXQ(i1, i1)]; // smaller index rules 
         Su[IDXQ(i2, i1)] = Su[IDXQ(i1, i1)]; // symmetrical
      }
    }

    // Mu is the only matrix we do in 2D, because we need to divide by it
    // We choose the negative of Matlab Mu to make it positive definite
    for ( i1 = 1; i1 <= q; i1++ ) 
      for ( i2 = i1; i2 <=q ; i2 += 2 )
        for ( i3 = 0; i3 <= q; i3++ ) {
          Mu[i1+i3*(q+1)+(i2+i3*(q+1))*(q+1)*(q+1)] += Mv[i3]*2.0*i1*(i1+1)/bL;
          Mu[i3+i1*(q+1)+(i3+i2*(q+1))*(q+1)*(q+1)] += Mv[i3]*2.0*i1*(i1+1)/bL;
        }
    Mu[0] = bL*bL; // special case equation (2.9)

    // We don't need Mu per se; we store only its Cholesky factorization
    ldMu = (q+1)*(q+1);
    dpotrf_(&uplo, &ldMu, Mu, &ldMu, &info);

    // Note that Sv is missing the special case equation (2.9)
    // We'll set the first row manually when we use Sv in the DAB stage.
    for ( i1 = 2; i1 <= q; i1++ )
      for ( i2 = i1-2; i2 >= 0; i2 -= 2 )
        Sv[IDXQ(i1, i2)] = -2.0*(i1+i2+1)*(i1-i2)/bL;

/* debug print Mu
for ( i4 = 0; i4 < q+1; i4++ )
  for ( i3 = 0; i3 < q+1; i3++ ) {
    printf("Mu(:, :, %d, %d)=[\n", i3, i4);
    for ( i1 = 0; i1 < q+1; i1++ ) {
      for ( i2 = 0; i2 < q+1; i2++ )
        printf("%9.4f", Mu[i2+i1*(q+1)+(i3+i4*(q+1))*(q+1)*(q+1)]);
      printf("\n");
    }
    printf("]\n");
  } */

  
    // Initialize DAB to be zero everywhere
    u = calloc((q+1)*(q+1)*(P+1)*boxes, sizeof(double));
    v = calloc((q+1)*(q+1)*(P+1)*boxes, sizeof(double));
    u_t = malloc((q+1)*(q+1)*(P+1)*boxes*sizeof(double));
    v_t = malloc((q+1)*(q+1)*(P+1)*boxes*sizeof(double));
    u_del = malloc((q+1)*(q+1)*(P+1)*boxes*sizeof(double));
    v_del = malloc((q+1)*(q+1)*(P+1)*boxes*sizeof(double));
    u_now = malloc((q+1)*(q+1)*(P+1)*boxes*sizeof(double));
    v_now = malloc((q+1)*(q+1)*(P+1)*boxes*sizeof(double));

// debug print statements
/*
printf("c=%9.4f L=%9.4f Tf=%9.4f m=%d m_ref=%d\n", c, L, Tf, m, m_ref);
printf("L_ref=%9.4f k=%9.4f ks=%9.4f substeps=%d h=%9.4f\n", L_ref, k, ks, substeps, h);
printf("bL=%9.4f a_d=%9.4f boxes=%d\n", bL,a_d, boxes);
printf("u_int=[\n");
for ( i1 = 0; i1 < m+ord+1; i1++ ) {
  for ( i2 = 0; i2 < m+ord+1; i2++ )
    printf("%0.16f,", u_int[IDXM(i2,i1)]);
  printf(";\n");
}
printf("]");
getchar();
*/

    t = 0; // current time
    while ( t < Tf - 1e-12 ) { // begin timestepping

      // advance interior with modified equation of order ord
      for ( i2 = ng; i2 < m+ng+1; i2++ ) // second spatial coordinate
        for ( i1 = ng; i1 < m+ng+1; i1++ ) { // first spatial coordinate
          up_int[IDXM(i1,i2)] = -um_int[IDXM(i1,i2)] + 2*u_int[IDXM(i1,i2)];
          for ( i3 = 2; i3 <= ord; i3 += 2 ) // number of derivatives total
            for ( i4 = 0; i4 <= i3; i4 += 2 ) { // number of first coordinate derivatives
              deriv_factor = 2*binom[i3/2][i4/2]*pow(c*k/h, i3)/factorial[i3];
              for ( i6 = 4-(ng-i4/2); i6 <= 4+(ng-i4/2); i6++ ) // second coordinate stencil point
                for ( i5 = 4-(ng-(i3-i4)/2); i5 <= 4+(ng-(i3-i4)/2); i5++ ) // first coordinate stencil point
                  up_int[IDXM(i1,i2)] += deriv_factor*cfdc[i4][ng-i3/2][i5]*cfdc[i3-i4][ng-i3/2][i6]*u_int[IDXM(i1-4+i5,i2-4+i6)];
            } // end number of first coordinate derivatives
         } // end first spatial coordinate

      // add artificial dissipation of order ord to interior
      for ( i2 = ng; i2 < m+ng+1; i2++ ) // second spatial coordinate
        for ( i1 = ng; i1 < m+ng+1; i1++ ) // first spatial coordinate
          for ( i5 = 4-ng; i5 <= 4+ng; i5++ ) // stencil point
            up_int[IDXM(i1,i2)] += pow(-1, ng-1)*a_d*c*c*k*cfdc[ord][0][i5]*(u_int[IDXM(i1+i5-4,i2)] \
              + u_int[IDXM(i1,i2+i5-4)] - um_int[IDXM(i1+i5-4,i2)] - um_int[IDXM(i1,i2+i5-4)]);

      // advance reference solution with modified equation of order ord
      for ( i2 = ng; i2 < m_ref+ng+1; i2++ ) // second spatial coordinate
        for ( i1 = ng; i1 < m+ng+1; i1++ ) { // first spatial coordinate
          up_ref[IDXM(i1,i2)] = -um_ref[IDXM(i1,i2)] + 2*u_ref[IDXM(i1,i2)];
          for ( i3 = 2; i3 <= ord; i3 += 2 ) // number of derivatives total
            for ( i4 = 0; i4 <= i3; i4 += 2 ) { // number of first coordinate derivatives
              deriv_factor = 2*binom[i3/2][i4/2]*pow(c*k/h, i3)/factorial[i3];
              for ( i6 = 4-(ng-i4/2); i6 <= 4+(ng-i4/2); i6++ ) // second coordinate stencil point
                for ( i5 = 4-(ng-(i3-i4)/2); i5 <= 4+(ng-(i3-i4)/2); i5++ ) // first coordinate stencil point
                  up_ref[IDXM(i1,i2)] += deriv_factor*cfdc[i4][ng-i3/2][i5]*cfdc[i3-i4][ng-i3/2][i6]*u_ref[IDXM(i1-4+i5,i2-4+i6)];
            } // end number of first coordinate derivatives
         } // end first spatial coordinate

      // reuse old edge u and u_x time derivatives rather than recalculating
      for ( i1 = 0; i1 < m+ord+1; i1++ ) { // first spatial coordinate
        for ( i2 = 0; i2 < ng+1; i2++ )
          u_deriv_right_grid[i2 + (ord+2)*i1] = u_deriv_right_grid[i2+ng+1 + (ord+2)*i1];
        for ( i2 = 0; i2 < ng; i2++ )
          u_x_deriv_right_grid[i2 + ord*i1] = u_x_deriv_right_grid[i2+ng + ord*i1];
      }

      // calculate current time derivatives of u on right edge grid points
      for ( i1 = ng; i1 < m+ng+1; i1++ ) { // first spatial coordinate (second is m+ng)
        u_deriv_right_grid[ng+1 + (ord+2)*i1] = u_int[IDXM(i1, m+ng)]; // zeroth derivative is value
        for ( i3 = 2; i3 <= ord; i3 += 2 ) { // number of derivatives total
          u_deriv_right_grid[ng+1+i3/2 + (ord+2)*i1] = 0;
          for ( i4 = 0; i4 <= i3; i4 += 2 ) { // number of first coordinate derivatives
            deriv_factor = binom[i3/2][i4/2]*pow(c/h, i3);
            for ( i6 = 4-(ng-i4/2); i6 <= 4+(ng-i4/2); i6++ ) // second coordinate stencil point
              for ( i5 = 4-(ng-(i3-i4)/2); i5 <= 4+(ng-(i3-i4)/2); i5++ ) // first coordinate stencil point
                u_deriv_right_grid[ng+1+i3/2 + (ord+2)*i1] += deriv_factor*cfdc[i4][ng-i3/2][i5]*cfdc[i3-i4][ng-i3/2][i6]*u_int[IDXM(i1-4+i5,m+ng-4+i6)];
          } // end number of first coordinate derivatives
        } // end number of derivatives total
      } // end first spatial coordinate

      // calculate current time derivatives of u_x on right edge grid points
      for ( i1 = ng; i1 < m+ng+1; i1++ ) { // first spatial coordinate (second is m+ng)
        for ( i3 = 1; i3 <= ord-1; i3 += 2 ) { // number of derivatives total
          u_x_deriv_right_grid[ng+i3/2 + ord*i1] = 0;
          for ( i4 = 0; i4 <= i3; i4 += 2 ) { // number of first coordinate derivatives
            deriv_factor = binom[i3/2][i4/2]*pow(c/h, i3)/c;
            for ( i6 = 4-(ng-i4/2); i6 <= 4+(ng-i4/2); i6++ ) // second coordinate stencil point
              for ( i5 = 4-(ng-(1+i3-i4)/2); i5 <= 4+(ng-(1+i3-i4)/2); i5++ ) // first coordinate stencil point
                u_x_deriv_right_grid[ng+i3/2 + ord*i1] += deriv_factor*cfdc[i4][ng-1-i3/2][i5]*cfdc[i3-i4][ng-1-i3/2][i6]*u_int[IDXM(i1-4+i5,m+ng-4+i6)];
          } // end number of first coordinate derivatives
        } // end number of derivatives total
      } // end first spatial coordinate

      // enforce Dirichlet boundaries above and below right edge
      // by odd extension of the time derivatives of u and u_x
      // or enforce periodic boundaries by copying values
      if ( periodic_ns )
        for ( i1 = 0; i1 < ng; i1++ ) {
          for ( i2 = ng+1; i2 < ord+2; i2++ ) {
            u_deriv_right_grid[i2 + (ord+2)*i1] = u_deriv_right_grid[i2 + (ord+2)*(m+i1)];
            u_deriv_right_grid[i2 + (ord+2)*(m+ng+1+i1)] = u_deriv_right_grid[i2 + (ord+2)*(ng+1+i1)];
          }
          for ( i2 = ng; i2 < ord; i2++ ) {
            u_x_deriv_right_grid[i2 + ord*i1] = u_x_deriv_right_grid[i2 + ord*(m+i1)];
            u_x_deriv_right_grid[i2 + ord*(m+ng+1+i1)] = u_x_deriv_right_grid[i2 + ord*(ng+1+i1)];
          }
        }
      else
        for ( i1 = 0; i1 < ng; i1++ ) {
          for ( i2 = ng+1; i2 < ord+2; i2++ ) {
            u_deriv_right_grid[i2 + (ord+2)*i1] = -u_deriv_right_grid[i2 + (ord+2)*(ord-i1)];
            u_deriv_right_grid[i2 + (ord+2)*(m+ord-i1)] = -u_deriv_right_grid[i2 + (ord+2)*(m+i1)];
          }
          for ( i2 = ng; i2 < ord; i2++ ) {
            u_x_deriv_right_grid[i2 + ord*i1] = -u_x_deriv_right_grid[i2 + ord*(ord-i1)];
            u_x_deriv_right_grid[i2 + ord*(m+ord-i1)] = -u_x_deriv_right_grid[i2 + ord*(m+i1)];
          }
        }

      // interpolate time derivatives of u on edge in space to lgl nodes
      for ( i1 = 0; i1 < boxes; i1++ ) { // which box
        for ( i2 = 0; i2 < ord+2; i2++ ) { // which time derivative
          // for the corner of the box, we simply copy the value
          u_deriv_right_lgl[i2 + (ord+2)*i1*q] = u_deriv_right_grid[i2 + (ord+2)*(i1*h_per_cell+ng)];
          // for the edge of the box, we interpolate
          for ( i3 = 0; i3 < q-1; i3++ ) { // which node of box
            u_deriv_right_lgl[i2 + (ord+2)*(i1*q + i3+1)] = 0;
            for ( i4 = 0; i4 < npilgl; i4++ ) // summation counter
              u_deriv_right_lgl[i2 + (ord+2)*(i1*q + i3+1)] += grid_to_lgl[i3 + (q-1)*i4] \
                *u_deriv_right_grid[i2 + (ord+2)*(i1*h_per_cell + i4)];
          }
        }
      }
      // one last grid point to copy
      for ( i2 = 0; i2 < ord+2; i2++ ) // which time derivative
        u_deriv_right_lgl[i2 + (ord+2)*boxes*q] = u_deriv_right_grid[i2 + (ord+2)*(boxes*h_per_cell+ng)];

      // interpolate time derivatives of u_x on edge in space to lgl nodes
      for ( i1 = 0; i1 < boxes; i1++ ) { // which box
        for ( i2 = 0; i2 < ord; i2++ ) { // which time derivative
          // for the corner of the box, we simply copy the value
          u_x_deriv_right_lgl[i2 + ord*i1*q] = u_x_deriv_right_grid[i2 + ord*(i1*h_per_cell+ng)];
          // for the edge of the box, we interpolate
          for ( i3 = 0; i3 < q-1; i3++ ) { // which node of box
            u_x_deriv_right_lgl[i2 + ord*(i1*q + i3+1)] = 0;
            for ( i4 = 0; i4 < npilgl; i4++ ) // summation counter
              u_x_deriv_right_lgl[i2 + ord*(i1*q + i3+1)] += grid_to_lgl[i3 + (q-1)*i4] \
                *u_x_deriv_right_grid[i2 + ord*(i1*h_per_cell + i4)];
          }
        }
      }
      // one last grid point to copy
      for ( i2 = 0; i2 < ord; i2++ ) // which time derivative
        u_x_deriv_right_lgl[i2 + ord*boxes*q] = u_x_deriv_right_grid[i2 + ord*(boxes*h_per_cell+ng)];

      // Hermite interpolate time derivatives into a polynomial in time
      nrhs = boxes*q + 1;
      dgetrs_(&trans, &ldHerm_u, &nrhs, Herm_u, &ldHerm_u, Herm_u_pivot, \
        u_deriv_right_lgl, &ldHerm_u, &info);
      if (info != 0)
        printf("Hermite interpolation of u failed with error code %d", info);

      dgetrs_(&trans, &ldHerm_u_x, &nrhs, Herm_u_x, &ldHerm_u_x, Herm_u_x_pivot, \
        u_x_deriv_right_lgl, &ldHerm_u_x, &info);
      if (info != 0)
        printf("Hermite interpolation of u_x failed with error code %d", info);
      
      // advance DAB
      for ( substep = 0; substep < substeps; substep++ ) {
        //reset RK variables
        for ( i1 = 0; i1 < (q+1)*(q+1)*(P+1)*boxes; i1++ ) {
          u_del[i1] = 0;
          v_del[i1] = 0;
          u_t[i1] = 0;
          v_t[i1] = 0;
        }

        for ( stage = 0; stage < 4; stage++ ) {
          // set stage values of u and v
          for ( i1 = 0; i1 < (q+1)*(q+1)*(P+1)*boxes; i1++ ) {
            u_now[i1] = u[i1] + ks*rktimes[stage]*u_t[i1];
            v_now[i1] = v[i1] + ks*rktimes[stage]*v_t[i1];
          }
          // t_now indicates how far we are from center of time polynomial
          // which is Hermite interpolated between the two previous interior times
          t_now = k/2.0 + ks*(substep + rktimes[stage]);
 
          // calculate u_t and u_x at lgl nodes at t_now
          for ( i1 = 0; i1 <= q*boxes; i1++ ) { // which node location
            u_t_interface_lgl[i1] = 0;
            for ( i2 = ord+1; i2 >= 2; i2-- ) { // which power of t_now
              u_t_interface_lgl[i1] += i2*u_deriv_right_lgl[i2 + (ord+2)*i1]; 
              u_t_interface_lgl[i1] *= t_now;
            }
            u_t_interface_lgl[i1] += u_deriv_right_lgl[1 + (ord+2)*i1]; 
            
            u_x_interface_lgl[i1] = 0;
            for ( i2 = ord-1; i2 >= 1; i2-- ) { // which power of t_now
              u_x_interface_lgl[i1] += u_x_deriv_right_lgl[i2 + ord*i1]; 
              u_x_interface_lgl[i1] *= t_now;
            }
            u_x_interface_lgl[i1] += u_x_deriv_right_lgl[ord*i1]; 
          }
          
          retval = DAB_stage( c, bL, q, P, boxes, Ao, Ai, Bo, Bi, Do, Di, \
            u_now, v_now, u_t, v_t, Mv, Su, Mu, Sv, ldMu, u_t_interface_lgl, \
            u_x_interface_lgl, basis_to_lgl, lgl_weight, periodic_ns );

          // accumulate change
          for ( i1 = 0; i1 < (q+1)*(q+1)*(P+1)*boxes; i1++ ) {
            u_del[i1] += ks*rkweights[stage]*u_t[i1];
            v_del[i1] += ks*rkweights[stage]*v_t[i1];
          }

        } // end stages

        // apply change
        for ( i1 = 0; i1 < (q+1)*(q+1)*(P+1)*boxes; i1++ ) {
          u[i1] += u_del[i1];
          v[i1] += v_del[i1];
        }

        t += ks; // advance time

      } // end substeps

      // Set ghost points outside east boundary of interior to zero
      for ( i2 = m+ng+1; i2 < m+ord+1; i2++ ) // second coordinate (normal)
        for ( i1 = ng; i1 < m+ng+1; i1++ ) // first coordinate (tangential)
          up_int[IDXM(i1,i2)] = 0;
      
      // Get ghost points outside east boundary of interior from DAB
      for ( i5 = 0; i5 < boxes; i5++ ) // box
        for ( i3 = 0; i3 <=q; i3++ ) // tangential basis function index
          for ( i4 = 0; i4 <=q; i4++ ) // normal basis function index
            for ( i2 = 0; i2 < ng; i2++ ) // second coordinate (normal)
              for ( i1 = 0; i1 <= h_per_cell; i1++ ) // first coordinate (tangential)
                up_int[IDXM(i1 + i5*h_per_cell+ng, m+ng+1+i2)] += basis_to_ghost[i1 + (h_per_cell+1)*i3] \
                  *basis_to_ghost[i2+1 + (h_per_cell+1)*i4]*u[IDXQPB(i4, i3, 0, i5)];
      // We double-counted the edges of boxes; halve (i.e. average) those
      for ( i5 = 1; i5 < boxes; i5++ ) // box
        for ( i2 = 0; i2 < ng; i2++ ) // second coordinate (normal)
          up_int[IDXM(i5*h_per_cell+ng, m+ng+1+i2)] /= 2.0;
      // If the north-south boundaries are periodic, average the ghost points
      if ( periodic_ns )
        for ( i2 = 0; i2 < ng; i2++ ) // second coordinate (normal)
          up_int[IDXM(ng, m+ng+1+i2)] = up_int[IDXM(m+ng, m+ng+1+i2)] = \
            (up_int[IDXM(ng, m+ng+1+i2)] + up_int[IDXM(m+ng, m+ng+1+i2)])/2.0;
        
      // west zero Dirichlet boundary of interior enforced by odd extension
      for ( i2 = 0; i2 < ng; i2++ ) // second coordinate (normal)
        for ( i1 = ng; i1 < m+ng+1; i1++ ) // first coordinate (tangential)
          up_int[IDXM(i1, i2)] = -up_int[IDXM(i1, ord-i2)];
      // north-south boundaries of interior either periodic or Dirichlet
      if ( periodic_ns )
        for ( i2 = 0; i2 < m+ord+1; i2++ ) // second coordinate (tangential)
          for ( i1 = 0; i1 < ng; i1++ ) { // first coordinate (normal)
            up_int[IDXM(i1, i2)] = up_int[IDXM(m+i1, i2)];
            up_int[IDXM(m+ng+1+i1, i2)] = up_int[IDXM(ng+1+i1, i2)];
          }
      else // Dirichlet
        for ( i2 = 0; i2 < m+ord+1; i2++ ) // second coordinate (tangential)
          for ( i1 = 0; i1 < ng; i1++ ) { // first coordinate (normal)
            up_int[IDXM(i1, i2)] = -up_int[IDXM(ord-i1, i2)];
            up_int[IDXM(m+ord-i1, i2)] = -up_int[IDXM(m+i1, i2)];
          }

      // east-west Dirichlet boundaries of reference solution enforced by odd extension
      for ( i2 = 0; i2 < ng; i2++ ) // second coordinate (normal)
        for ( i1 = ng; i1 < m+ng+1; i1++ ) { // first coordinate (tangential)
          up_ref[IDXM(i1, i2)] = -up_ref[IDXM(i1, ord-i2)];
          up_ref[IDXM(i1, m_ref+ord-i2)] = -up_ref[IDXM(i1, m_ref+i2)];
        }
      // north-south boundaries of reference solution either periodic or Dirichlet
      if ( periodic_ns )
        for ( i2 = 0; i2 < m_ref+ord+1; i2++ ) // second coordinate (tangential)
          for ( i1 = 0; i1 < ng; i1++ ) { // first coordinate (normal)
            up_ref[IDXM(i1, i2)] = up_ref[IDXM(m+i1, i2)];
            up_ref[IDXM(m+ng+1+i1, i2)] = up_ref[IDXM(ng+1+i1, i2)];
          }
      else // Dirichlet
        for ( i2 = 0; i2 < m_ref+ord+1; i2++ ) // second coordinate (tangential)
          for ( i1 = 0; i1 < ng; i1++ ) { // first coordinate (normal)
            up_ref[IDXM(i1,i2)] = -up_ref[IDXM(ord-i1,i2)];
            up_ref[IDXM(m+ord-i1,i2)] = -up_ref[IDXM(m+i1,i2)];
          }

      // swap array pointers for interior
      temp = um_int;
      um_int = u_int;
      u_int = up_int;
      up_int = temp;

      // swap array pointers for reference solution
      temp = um_ref;
      um_ref = u_ref;
      u_ref = up_ref;
      up_ref = temp;

    } // end timestepping

    // calculate discrepancy between u_int and u_ref
    err = 0;
    for ( i2 = ng; i2 < m+ng+1; i2++)
      for ( i1 = ng; i1 < m+ng+1; i1++) {
        foo = fabs(u_int[IDXM(i1, i2)] - u_ref[IDXM(i1, i2)]);
//        foo = fabs(u_int[IDXM(i1, i2)]); // maximum of interior
//        foo = fabs(u_ref[IDXM(i1, i2)]); // maximum of reference solution
        if ( foo > err ) {
          err = foo;
          max_err_loc[0] = i1;
          max_err_loc[1] = i2;
        }
      }

    // print error
    printf( "For m = %d, P = %d, q = %d, max abs error = %e at coordinates (%d,%d)\n", \
      m, P, q, err, max_err_loc[0], max_err_loc[1] );
    if ( old_err != 0 )
      printf( "Apparent convergence order: %f\n", \
        log(old_err/err)/log((double)m/(m-refine_step)) );
    old_err = err;

    free(lgl_loc);
    free(lgl_weight);
    free(grid_to_lgl);
    free(basis_to_lgl);
    free(Herm_u);
    free(Herm_u_pivot);
    free(Herm_u_x);
    free(Herm_u_x_pivot);
    free(u_deriv_right_grid);
    free(u_x_deriv_right_grid);
    free(u_deriv_right_lgl);
    free(u_x_deriv_right_lgl);
    free(u_t_interface_lgl);
    free(u_x_interface_lgl);
    free(u_int);
    free(um_int);
    free(up_int);
    free(u_ref);
    free(um_ref);
    free(up_ref);
    free(u);
    free(v);
    free(u_t);
    free(v_t);
    free(u_del);
    free(v_del);
    free(u_now);
    free(v_now);
    free(Mu);
    free(Mv);
    free(Su);
    free(Sv);

  } // end refinement

  free(Ao);
  free(Ai);
  free(Bo);
  free(Bi);
  free(Do);
  free(Di);

//} // end q
//} // end P
  return(0);
}
