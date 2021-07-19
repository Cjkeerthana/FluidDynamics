#include <mpi.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>


#ifndef M_PI
  #define M_PI 3.1415926535897932
#endif

int main(int argc, char* argv[])
{
 double domain_length=2.0, velocity=1.0, diffCoeff=0.05, waveNo=1.0, time=0.5, courant=0.3; 
 double dt, dx, nstep;
 size_t N=201, res=0, offset=0, N_local;
 int rank, np;

 MPI_Status status;
 MPI_Init(&argc &argv);
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 MPI_Comm_size(MPI_COMM_WORLD, &np);
 
 if( np < 1){
  fprintf(stderr, "Usage : mpi -np n %s number of iterations", argv[1]);
  MPI_Finalize();
  exit(-1);
 }
  
 if(argc == 8){
  N = atoll(argv[1]) + 1;
  domain_length = atoll(argv[2]);
  velocity = atoll(argv[3]);
  diffCoeff = atoll(argv[4]);
  waveNo = atoll(argv[5]);
  time = atoll(argv[6]);
  courant = atoll(argv[7]);
 }
 
 dx = domain_length/(N-1);
 dt = courant*dx*dx/diffCoeff;
 nstep = time/dt;
 
 

 size_t byte_dimension = sizeof(double) * (N+2);
 double* x_local = (double*)calloc(0, byte_dimension);
 double* f_local = (double*)calloc(0, byte_dimension);
 double* fo_local = (double*)calloc(0, byte_dimension);

 N_local = N/np;
 res = N%np;
 if(rank < res)N_local++;

 for(size_t i=0; i<N_local; i++){ 	
   i_glob = i + N_local*rank + offset;
   x_local[i] = dx*i_glob; 
   fo_local[i] = 0.5*sin(2*M_PI*waveNo*x_local[i]); 
   f_local[i] = fo_local[i];
 }
 
 
}
