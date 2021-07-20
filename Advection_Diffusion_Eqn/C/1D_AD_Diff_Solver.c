#include <mpi.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>


#ifndef M_PI
  #define M_PI 3.1415926535897932
#endif

void exchange_borders(double* fo_local, size_t N_local, int prev, int next, MPI_Request **request);
void evolve_FTCS(double* f_local, double* fo_local, size_t row_start, size_t row_end, double conv, double diff);
void swap(double** arr1, double** arr2, double* tmp);
void save_gnuplot(FILE* file, double* arr, double* x, size_t start, size_t end);

int main(int argc, char* argv[])
{
 double domain_length=2.0, velocity=1.0, diffCoeff=0.05, waveNo=1.0, time=0.5, courant=0.25; 
 double dt, dx;
 size_t N=200, res=0, offset=0, N_local, nstep, i_glob;
 int rank, np;

 MPI_Init(&argc, &argv);
 MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 MPI_Comm_size(MPI_COMM_WORLD, &np);
 
 if( np < 1){
  fprintf(stderr, "Usage : mpi -np n %s number of iterations", argv[1]);
  MPI_Finalize();
  exit(-1);
 }
  
 if(rank == 0){
  FILE* infile = fopen("input.txt", "r");
  if(infile){
   int stat = fscanf(infile, "%*s %ld", &N);
   stat = fscanf(infile, "%*s %lf", &domain_length);
   stat = fscanf(infile, "%*s %lf", &velocity);
   stat = fscanf(infile, "%*s %lf", &diffCoeff);
   stat = fscanf(infile, "%*s %lf", &waveNo);
   stat = fscanf(infile, "%*s %lf", &time);
   stat = fscanf(infile, "%*s %lf", &courant);
   fclose(infile);
  }
  else{
   printf("Input file missing, please provide input file\n");
   exit(-1);
  }
 }

 MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
 MPI_Bcast(&domain_length, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 MPI_Bcast(&velocity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 MPI_Bcast(&diffCoeff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 MPI_Bcast(&waveNo, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 MPI_Bcast(&time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
 MPI_Bcast(&courant, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

 dx = domain_length/N;
 dt = courant*dx*dx/diffCoeff;
 nstep = (size_t) ceil(time/dt);
 
 double conv = 0.5*velocity*dt/dx;
 double diff = diffCoeff*dt/(dx*dx);

 N_local = N/np;
 res = N%np;
 if(rank < res)N_local++;
 if(rank >= res)offset = res;

 size_t byte_dimension = sizeof(double) * (N_local+2);
 double* x_local = (double*)malloc(byte_dimension);
 double* f_local = (double*)malloc(byte_dimension);
 double* fo_local = (double*)malloc(byte_dimension);
 double* tmp_array;

 for(size_t i=1; i<=N_local; i++){ 	
   i_glob = i + N_local*rank + offset;
   x_local[i] = dx*i_glob; 
   fo_local[i] = 0.5*sin(2*M_PI*waveNo*x_local[i]);
   f_local[i] = fo_local[i];
 }

 if(rank != 0){
 x_local[0] = x_local[1];
 fo_local[0] = fo_local[1];
 f_local[0] = f_local[1];
 }
 
 if(rank != np-1){ 
 x_local[N_local+1] = x_local[N_local];
 fo_local[N_local+1] = fo_local[N_local];
 f_local[N_local+1] = f_local[N_local];
 }
 else {
  x_local[N_local+1]=domain_length;
  fo_local[N_local+1] = 0.5*sin(2*M_PI*waveNo*domain_length);
  f_local[N_local+1] = fo_local[N_local+1];
 }

 int prev = rank-1;
 int next = rank+1;
 
 if(rank == 0) prev=np-1;
 if(rank == np-1)next=0;

 MPI_Status status[2];
 MPI_Request* request;
 request = (MPI_Request*)malloc(2*sizeof(MPI_Request));

 for(size_t it=0; it<=nstep; it++){
   exchange_borders(fo_local, N_local, prev, next, &request);
   evolve_FTCS(f_local, fo_local, 2, N_local, conv, diff);
   MPI_Waitall(2, request, status);
   evolve_FTCS(f_local, fo_local, 1, 2, conv, diff);
   evolve_FTCS(f_local, fo_local, N_local, N_local+1, conv, diff);
   swap(&f_local, &fo_local, tmp_array);
 }

 if(rank != 0){
   MPI_Send(f_local+1, N_local, MPI_DOUBLE, 0, 300, MPI_COMM_WORLD);
   MPI_Send(x_local+1, N_local, MPI_DOUBLE, 0, 400, MPI_COMM_WORLD);
 }
 else{
  FILE* file;
  file = fopen("solution.dat","w");
  save_gnuplot(file, f_local, x_local, 0, N_local+1); 
  for(int i=1; i<np; i++){
   if(i == res) N_local--;
   MPI_Recv(f_local+1, N_local, MPI_DOUBLE, i, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   MPI_Recv(x_local+1, N_local, MPI_DOUBLE, i, 400, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   save_gnuplot(file, f_local, x_local, 1, N_local+1);
  }
  fclose(file);
 }

 if(rank == 0){
  double* x = (double*)malloc(sizeof(double)*(N+1));
  double* f = (double*)malloc(sizeof(double)*(N+1));
  double* fo = (double*)malloc(sizeof(double)*(N+1));

  for(size_t i=0; i<N+1; i++){
    x[i] = dx*i; 
    fo[i] = 0.5*sin(2*M_PI*waveNo*x[i]);
    f[i] = fo[i];
  }
  
  for(size_t it=0; it<1000; it++){
    evolve_FTCS(f, fo, 1, N, conv, diff);
    f[N] = fo[N] - conv*(fo[1]-fo[N-1]) + diff*(fo[1]-2*fo[N]+fo[N-1]);
    f[0] = f[N];
    swap(&f, &fo, tmp_array);
  }
  FILE* file;
  file = fopen("solution_serial.dat","w");
  save_gnuplot(file, f, x, 0, N+1);
  fclose(file);
 }


 MPI_Finalize();
 return 0;
}

void exchange_borders(double* fo_local, size_t N_local, int prev, int next, MPI_Request **request){
 MPI_Isend(fo_local+1, 1, MPI_DOUBLE, prev, 100, MPI_COMM_WORLD, *request);
 MPI_Isend(fo_local+(N_local), 1, MPI_DOUBLE, next, 200, MPI_COMM_WORLD, *request+1);
 MPI_Irecv(fo_local, 1, MPI_DOUBLE, prev, 200, MPI_COMM_WORLD, *request);
 MPI_Irecv(fo_local+(N_local+1), 1, MPI_DOUBLE, next, 100, MPI_COMM_WORLD, *request+1);
}

void evolve_FTCS(double* f_local, double* fo_local, size_t row_start, size_t row_end, double conv, double diff){
for(size_t i=row_start; i<row_end; i++){
 f_local[i] = fo_local[i] - conv*(fo_local[i+1]-fo_local[i-1]) + diff*(fo_local[i+1]-2*fo_local[i]+fo_local[i-1]);
}
}

void swap(double** arr1, double** arr2, double* tmp){
 tmp = *arr1;
 *arr1 = *arr2;
 *arr2 = tmp;
}

void save_gnuplot(FILE* file, double* arr, double* x, size_t start, size_t end){
 for(size_t i=start; i<end; i++){
   fprintf(file, "%f\t%f\n", x[i], arr[i]);
 }
}

