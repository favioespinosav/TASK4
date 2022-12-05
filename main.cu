#include <iostream>
#include <vector>
#include <fstream>

#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <mpi.h>
//#include <omp.h>

using namespace std;

double EPS = 0.00002;
double my_temp = 0.5;

double A1=-1, A2=2, B1=-2, B2=2;
int d(int t ,int k ){
        return int(t/k);
    }
__device__ double k_c(double x, double y){
    return 4.0 + x + y;
}
__device__ double a_c(int i, int j,double *x,double *y,double h1,double h2)
{
    return k_c(x[i] - 0.5 * h1, y[j]);
}

__device__ double w_x_c(double * &w, int i, int j,int begin_i,int end_i,int begin_j ,int end_j,double *recieve_bottom,double *recieve_top,int N, int M,int sv_n,int sv_m,double h1,double h2)
{
    if (begin_i > 0 && i == begin_i)
        return (w[i * (N+1) + j] - recieve_top[j % sv_n]) / h1;
    if (end_i < M && i == end_i + 1)
        return (recieve_bottom[j % sv_n] - w[(i-1) * (N+1) + j]) / h1;
    return   (w[i * (N+1) + j] - w[(i-1) * (N+1) + j]) / h1;
}

__device__ double w_y_c(double * &w, int i, int j,int begin_i,int end_i,int begin_j ,int end_j,double *recieve_right,double *recieve_left,int N, int M,int sv_n,int sv_m,double h1,double h2)
{
    if (begin_j > 0 && j == begin_j) //левый край блока
        return (w[i * (N+1) + j]  - recieve_left[i % sv_m]) / h2;
    if (end_j < N && j == end_j + 1) // правый край блока
        return (recieve_right[i % sv_m] - w[i * (N+1) + j-1]) / h2;
    return (w[i * (N+1) + j] - w[i * (N+1) + j-1] ) / h2;
}

__device__ double q_c(double x, double y)
{
        return (x+y)*(x+y);
}
__device__ double b_c(int i, int j,double *x,double *y,double h1, double h2)
{
    return k_c(x[i], y[j] - 0.5 * h2);
}

__device__ double b_wy_c(double * &w, int i, int j,int begin_i,int end_i, int begin_j, int end_j,int M, int  N,double * x,double * y,double h1,double h2,double *recieve_bottom,double *recieve_top,double *recieve_right,double *recieve_left,int sv_n,int sv_m)
{
    return 1.0 * (b_c(i, j + 1,x,y,h1,h2) * w_y_c(w, i, j + 1,begin_i,end_i,begin_j,end_j,recieve_right,recieve_left,N,M,sv_n,sv_m, h1, h2) - b_c(i, j,x,y,h1,h2) * w_y_c(w, i, j,begin_i,end_i,begin_j,end_j,recieve_right,recieve_left,N,M,sv_n,sv_m, h1, h2)) / h2;
}
__device__ double a_wx_c(double * &w, int i, int j,int begin_i,int end_i, int begin_j, int end_j,int M, int  N,double * x,double * y,double h1,double h2,double *recieve_bottom,double *recieve_top,double *recieve_right,double *recieve_left,int sv_n,int sv_m)
{
   return 1.0 * (a_c(i + 1, j,x,y,h1,h2) * w_x_c(w, i + 1, j,begin_i,end_i,begin_j,end_j,recieve_bottom,recieve_top,N,M,sv_n,sv_m,h1,h2) - a_c(i, j,x,y,h1,h2) * w_x_c(w, i, j,begin_i,end_i,begin_j,end_j,recieve_bottom,recieve_top,N,M,sv_n,sv_m, h1, h2)) / h1;

}
__device__ double oper_laplass_c(double * &w, int i, int j,int begin_i,int end_i, int begin_j, int end_j,int M, int  N,double * x,double * y,double h1,double h2,double *recieve_bottom,double *recieve_top,double *recieve_right,double *recieve_left,int sv_n,int sv_m)
{
    return a_wx_c(w, i, j,begin_i,end_i,begin_j,end_j,M,N,x,y,h1,h2,recieve_bottom,recieve_top,recieve_right,recieve_left,sv_n,sv_m) + b_wy_c(w, i, j,begin_i,end_i,begin_j,end_j,M,N,x,y,h1,h2,recieve_bottom,recieve_top,recieve_right,recieve_left,sv_n,sv_m);
}

__global__ void cuda_A(double * w,int begin_i,int end_i, int begin_j, int end_j,int M, int  N,double   * A,double * x,double * y,double h1,double h2,double *recieve_bottom,double *recieve_top,double *recieve_right,double *recieve_left,int sv_n,int sv_m)
{
    static const int alpha_R = 0,
              alpha_L = 0,
              alpha_T = 0;
    int i = blockIdx.y * blockDim.y + threadIdx.y+begin_i;
    int j = blockIdx.x * blockDim.x + threadIdx.x+begin_j;

    if(i<= end_i && i<=M && j <= end_j && j<=N){

            if (i == 0 && j == N)
            {

                A[i * (N+1) + j] = -2. / h1 * a_c(i+1, j, x, y, h1, h2) * w_x_c(w, i+1, j,begin_i, end_i, begin_j, end_j, recieve_bottom, recieve_top, N, M,sv_n, sv_m, h1, h2)  +2. / h2 * b_c(i, j, x, y, h1, h2) * w_y_c(w, i, j, begin_i, end_i, begin_j, end_j, recieve_right, recieve_left, N, M, sv_n, sv_m, h1, h2) + (q_c(x[i],y[j]) + 2. * alpha_L / h1 + 2. * alpha_T / h2) * w[i * (N+1) + j];
            }
            else if (i == M && j == N)
            {
                A[i * (N+1) + j] = 2. / h1 * a_c(i, j, x, y, h1, h2) * w_x_c(w, i, j,begin_i, end_i, begin_j, end_j, recieve_bottom, recieve_top, N, M, sv_n, sv_m, h1, h2)+2. / h2 * b_c(i, j, x, y, h1, h2) * w_y_c(w, i, j, begin_i, end_i, begin_j, end_j, recieve_right, recieve_left, N, M, sv_n, sv_m, h1, h2)+ (q_c(x[i],y[j]) + 2. * alpha_R / h1 + 2. * alpha_T / h2) * w[i * (N+1) + j];
            }
            else if (i == 0)
            {
                A[i * (N+1) + j] = - 2. / h1 *  a_c(i+1, j, x, y, h1, h2) * w_x_c(w, i+1, j,begin_i, end_i, begin_j, end_j, recieve_bottom, recieve_top, N, M,
                                          sv_n, sv_m, h1, h2)
                                        + (q_c(x[i],y[j]) + 2. * alpha_L / h1) * w[i * (N+1) + j]
                                        - b_wy_c(w, i, j, begin_i, end_i, begin_j, end_j, M, N, x, y, h1, h2, recieve_bottom,
                                              recieve_top, recieve_right, recieve_left, sv_n, sv_m);
            }
            else if (i == M)
            {
                A[i * (N+1) + j] = 2. / h1 * (a_c(i, j, x, y, h1, h2) * w_x_c(w, i, j,begin_i, end_i, begin_j, end_j, recieve_bottom, recieve_top, N, M,
                                          sv_n, sv_m, h1, h2))
                                        + (q_c(x[i],y[j]) + 2. * alpha_R / h1) * w[i * (N+1) + j]
                                        - b_wy_c(w, i, j, begin_i, end_i, begin_j, end_j, M, N, x, y, h1, h2, recieve_bottom,
                                              recieve_top, recieve_right, recieve_left, sv_n, sv_m);
            }
            else if (j == N)
            {
                A[i * (N+1) + j] = 2. / h2 *b_c(i, j, x, y, h1, h2) * w_y_c(w, i, j, begin_i, end_i, begin_j, end_j, recieve_right, recieve_left, N, M, sv_n, sv_m, h1, h2)
                                        + (q_c(x[i],y[j]) + 2. * alpha_T /h2) * w[i * (N+1) + j]
                                        - a_wx_c(w, i, j, begin_i, end_i, begin_j, end_j, M, N, x, y, h1, h2, recieve_bottom,recieve_top, recieve_right, recieve_left, sv_n, sv_m);
            }
            else
            {
                A[i * (N+1) + j] = -oper_laplass_c(w, i, j, begin_i, end_i, begin_j, end_j, M, N, x, y, h1, h2, recieve_bottom,
                                            recieve_top, recieve_right, recieve_left, sv_n, sv_m) + q_c(x[i],x[j]) * w[i * (N+1) + j];
            }

        }
    }
class NumMethod
{

    static const int alpha_R = 0,
              alpha_L = 0,
              alpha_T = 0,
              alpha_B = 0;

    int rank, size, begin_i, end_i, begin_j ,end_j, size_i, size_j;
    int M, N, sv_n, sv_m;
    double h1, h2;

    double * x;
    double * y;

    double * B;

    static const int tag = 608;
    static const int tag2 = 6081;
    double * recieve_top;
    double * recieve_bottom;
    double *  send_top;
    double * send_bottom;
    double * recieve_left;
    double * recieve_right;
    double * send_left;
    double * send_right;

    double u_fun(double x, double y)
    {
        return exp(1.-pow(x+y,2));
    }

    double k(double x, double y)
    {
        return 4.0 + x + y;
    }

    double q(double x, double y){
        return pow(x+y,2);
    }

    double F(double x, double y)
    {
        return exp(1.-pow(x+y,2))*(2.*(x+y)+4.*(x+4.)+pow(x+y,2)-8.*(x+4.)*pow(x+y,2));
    }

    double PSI_R(double y, double alpha = 1)
    {

        return -12*(2+y)*exp(1-pow(2+y,2));
    }

    double PSI_L(double y, double alpha = 0)
    {

         return 6*(y-1)*exp(1-pow(y-1,2));
    }

    double PSI_T(double x,  double alpha = 1)
    {
        double y = 2.;
        return -2*(x+2)*exp(1-pow(x+2,2))*q(x,y);
    }

    double PSI_B(double x, double alpha = 0)
    {
        double y = -2.;
        return exp(1-pow(x+y,2));
    }

    void B_fun()
    {

        for (int i = begin_i; i <= min(M, end_i); i++)
        {
            for (int j = begin_j; j <= min(N,end_j); j++)
            {

                //----------------CENTER--------------------
                if (j >= 1 && j <= N-1 && i >= 1 && i <= M-1)
                    B[i * (N+1) + j] = F(x[i],y[j]);

                //----------------LIMITS--------------------
                //RIGHT
                else if (j >= 1 && j <= N-1 && i == M)
                    B[M * (N+1) + j]  = F(x[M],y[j]) + 2. / h1 * PSI_R(y[j]);
                //LEFT
                else if (j >= 1 && j <= N-1 && i == 0)
                    B[0 * (N+1) + j]  = F(x[0],y[j]) + 2. / h1 * PSI_L(y[j]);
                //BOTTOM
                else if (i >= 1 && i <= M-1 && j == 0)
                    B[i * (N+1) + 0] = F(x[i],y[0]) + 2. / h2 * PSI_B(x[i]);
                //TOP
                else if (i >= 1 && i <= M-1 && j == N)
                    B[i * (N+1) + N] = F(x[i],y[N]) + 2. / h2 * PSI_T(x[i]);

                //----------------CORNERS-------------------
                //BOTTOM-LEFT P(0,0)
                else if(i == 0 && j == 0)
                    B[0 * (N+1) + 0] = F(x[0],y[0]) + 2. / h1 * PSI_L(y[0]) + 2. / h2 * PSI_B(x[0]);
                //BOTTOM-RIGHT P(M,0)
                else if (i == M && j == 0)
                    B[M * (N+1) + 0] = F(x[M],y[0]) + 2. / h1 * PSI_R(y[0]) + 2. / h2 * PSI_B(x[M]);
                //LEFT-TOP P(0,N)
                else if(i == 0 && j == N)
                    B[0 * (N+1) + N] = F(x[0],y[N]) + 2. / h1 * PSI_L(y[N]) + 2. / h2 * PSI_T(x[0]);
                //TOP-RIGHT P(M,N)
                else if (i == M && j == N)
                    B[M * (N+1) + N] = F(x[M],y[N]) + 2. / h1 * PSI_R(y[N]) + 2. / h2 * PSI_T(x[M]);



            }
        }
    }


    double rho(int i, int j)
    {
        double rho_1 = (i >= 1 && i <= M-1) ? 1.0 : 0.5;
        double rho_2 = (j >= 1 && j <= N-1) ? 1.0 : 0.5;
        return rho_1 * rho_2;
    }
    double scal_prod(double * &u, double * &v)
    {
        double sum = 0;
        double full_threads_sum;

        for(int i = begin_i; i <= min(M, end_i); i++)
        {
            double sum2 = 0;
            for (int j = begin_j; j <= min(N, end_j); j++)
            {
                if(j>0){
                sum2 += h2 * rho(i, j) *  u[i * (N+1) + j] * v[i * (N+1) + j];
                }
            }
            sum += h1 * sum2;
        }

        MPI_Allreduce(&sum, &full_threads_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return full_threads_sum;
    }

    double norma(double * &u)
    {
        return sqrt(scal_prod(u,u));
    }

    double w_x(double * &w, int i, int j)
    {

		if (begin_i > 0 && i == begin_i)
			return (w[i * (N+1) + j] - recieve_top[j % sv_n]) / h1;
		if (end_i < M && i == end_i + 1)
			return (recieve_bottom[j % sv_n] - w[(i-1) * (N+1) + j]) / h1;
        return   (w[i * (N+1) + j] - w[(i-1) * (N+1) + j]) / h1;
    }
    double w_y(double * &w, int i, int j)
    {
		if (begin_j > 0 && j == begin_j)
			return (w[i * (N+1) + j]  - recieve_left[i % sv_m]) / h2;
		if (end_j < N && j == end_j + 1)
			return (recieve_right[i % sv_m] - w[i * (N+1) + j-1]) / h2;
        return (w[i * (N+1) + j] - w[i * (N+1) + j-1] ) / h2;
    }

    double a(int i, int j)
    {
        return k(x[i] - 0.5 * h1, y[j]);
    }

    double b(int i, int j)
    {
        return k(x[i], y[j] - 0.5 * h2);
    }

    double a_wx(double * &w, int i, int j)
    {
        return 1.0 * (a(i + 1, j) * w_x(w, i + 1, j) - a(i, j) * w_x(w, i, j)) / h1;
    }

    double b_wy(double * &w, int i, int j)
    {
        return 1.0 * (b(i, j + 1) * w_y(w, i, j + 1) - b(i, j) * w_y(w, i, j)) / h2;
    }

    double oper_laplass(double * &w, int i, int j)
    {
        return a_wx(w, i, j) + b_wy(w, i, j);

    }

    double * matrix_Aw(double * &w){

        int N_size = min(N, end_j) - begin_j + 1;

        int M_size = min(M, end_i) - begin_i + 1;
        int rank_i = rank / size_j;
        int rank_j = rank % size_j;

        MPI_Request requests_j[4];
        MPI_Request requests_i[4];

        if (rank_i > 0)
        {

            for (int j = begin_j; j <= min(N, end_j); j++)
            {
                send_top[j % sv_n] = w[begin_i * (N+1) + j];
            }
            MPI_Isend(&send_top[0], sv_n, MPI_DOUBLE, rank - size_j, tag, MPI_COMM_WORLD, &requests_i[0]);
            MPI_Irecv(&recieve_top[0], sv_n, MPI_DOUBLE, rank - size_j, tag, MPI_COMM_WORLD, &requests_i[1]);
        }
        if (rank_i < size_i-1)
        {

            for (int j = begin_j; j <= min(N, end_j); j++)
                send_bottom[j % sv_n] = w[end_i * (N+1) + j];
            MPI_Isend(&send_bottom[0], sv_n, MPI_DOUBLE, rank + size_j, tag, MPI_COMM_WORLD, &requests_i[2]);
            MPI_Irecv(&recieve_bottom[0], sv_n, MPI_DOUBLE, rank + size_j, tag, MPI_COMM_WORLD, &requests_i[3]);
        }

        if (rank_i > 0 && rank_i < size_i - 1)
        {
            MPI_Waitall(4, requests_i, MPI_STATUSES_IGNORE);
        }
        if (rank_i > 0)
        {
            MPI_Waitall(2, requests_i, MPI_STATUSES_IGNORE);
        }
        if (rank_i < size_i - 1)
        {
            MPI_Waitall(2, &requests_i[2], MPI_STATUSES_IGNORE);
        }


        if (rank_j > 0)
        {

            for (int i = begin_i; i <= min(M, end_i); i++)
                send_left[i % sv_m] = w[i * (N+1) + begin_j];
            MPI_Isend(&send_left[0], sv_m, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD, &requests_j[0]);
            MPI_Irecv(&recieve_left[0], sv_m, MPI_DOUBLE, rank-1, tag, MPI_COMM_WORLD, &requests_j[1]);
        }

        if (rank_j < size_j-1)
        {

            for (int i = begin_i; i <= min(M, end_i); i++)
                send_right[i % sv_m] = w[i * (N+1) + end_j];
            MPI_Isend(&send_right[0], sv_m, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &requests_j[2]);
            MPI_Irecv(&recieve_right[0], sv_m, MPI_DOUBLE, rank+1, tag, MPI_COMM_WORLD, &requests_j[3]);
        }

        if (rank_j > 0 && rank_j < size_j - 1)
        {
            MPI_Waitall(4, requests_j, MPI_STATUSES_IGNORE);
        }
        if (rank_j > 0 )
        {
            MPI_Waitall(2, requests_j, MPI_STATUSES_IGNORE);
        }
        if (rank_j < size_j - 1)
        {
            MPI_Waitall(2, &requests_j[2], MPI_STATUSES_IGNORE);
        }

        // double * A((M + 1)*(N + 1),0);
        double * A;
        double *tmp_A;
        double *w_tmp,*tmp_recvbuf_next_i,*tmp_recvbuf_prev_i,*tmp_recvbuf_next_j,*tmp_recvbuf_prev_j,*tmp_x,*tmp_y;

        A = new double[(M + 1) * (N + 1)];
        for(int i = 0;i<=M;i++ ){
          for(int j = 0;j<=N;j++ ){
            A[i * (N + 1)+j] = 0;

            }

        }
        cudaMalloc(&tmp_A, (N+1)*(M+1)*sizeof(double));

        cudaMalloc(&w_tmp, (N+1)*(M+1)*sizeof(double));
        cudaMalloc(&tmp_y, (N+1)*sizeof(double));
        cudaMalloc(&tmp_x, (M+1)*sizeof(double));
        cudaMalloc(&tmp_recvbuf_next_i, sv_m*sizeof(double));
        cudaMalloc(&tmp_recvbuf_prev_i, sv_m*sizeof(double));
        cudaMalloc(&tmp_recvbuf_next_j, sv_n*sizeof(double));
        cudaMalloc(&tmp_recvbuf_prev_j, sv_n*sizeof(double));

        cudaMemcpy(tmp_A, A, (N+1)*(M+1)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(w_tmp, w, (N+1)*(M+1)*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(tmp_x, x, (M+1)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_y, y, (N+1)*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(tmp_recvbuf_next_i, recieve_bottom, sv_m*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_recvbuf_prev_i, recieve_top, sv_m*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_recvbuf_next_j, recieve_right, sv_n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(tmp_recvbuf_prev_j, recieve_left, sv_n*sizeof(double), cudaMemcpyHostToDevice);
        int dim = 16;
        dim3 threadsPerBlock(dim, dim);
        dim3 numBlocks((M_size) /  threadsPerBlock.x, (N_size)/  threadsPerBlock.y);
        cuda_A<<<numBlocks, threadsPerBlock>>>(w_tmp,begin_i,end_i,begin_j, end_j,M,N,tmp_A, tmp_x, tmp_y,h1, h2,tmp_recvbuf_next_i,tmp_recvbuf_prev_i,tmp_recvbuf_next_j,tmp_recvbuf_prev_j,sv_n,sv_m);
//cuda_A<<<numBlocks, threadsPerBlock>>>(w_tmp,begin_i,end_i,begin_j, end_j,M,N,tmp_A, tmp_x, tmp_y,h1, h2,tmp_recvbuf_next_i,tmp_recvbuf_prev_i,tmp_recvbuf_next_j,tmp_recvbuf_prev_j,sv_n,sv_m);
        cudaDeviceSynchronize();
        cudaMemcpy(A, tmp_A, (N+1)*(M+1)*sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(w_tmp);

        cudaFree(tmp_A);

        cudaFree(tmp_recvbuf_next_i);
        cudaFree(tmp_recvbuf_prev_i);
        cudaFree(tmp_recvbuf_next_j);
        cudaFree(tmp_recvbuf_prev_j);
        return A;
    }

    double tau(double *  &w)
    {
        double * Aw;
        Aw = new double [(M + 1) * (N + 1)];
        Aw = matrix_Aw(w);
        double norm = norma(Aw);
        return scal_prod(Aw,w) / (norm * norm);
    }

    double * r_k(double *  &w)
    {
        double *  Aw;
        Aw = new double [(M + 1) * (N + 1)];
        Aw =  matrix_Aw(w);
        for (int i = begin_i; i <= min(M, end_i); i++)
        {
            for (int j = begin_j; j <= min(N, end_j); j++)
            {
               // if(j>0){

                Aw[i * (N + 1) + j] = Aw[i * (N + 1) + j] - B[i * (N + 1) + j];
               // }
            }
        }
        return Aw;
    }

    int Iteration(double EPS)
    {
        double * r_prev;
        r_prev = new double[(M + 1) * (N + 1)];
        double * check_stop;
        check_stop = new double [(M + 1) * (N + 1)];
        r_prev = r_k(w);
        double tau_k_pus_1 = tau(r_prev) * my_temp;

        for (int i = begin_i; i <= min(M, end_i); i++)
        {
            for (int j = begin_j; j <= min(N, end_j); j++)
            {

                check_stop[i * (N+1) + j] = w[i * (N+1) + j];
                if(j>0){
                    w[i * (N+1) + j] = w[i * (N+1) + j] - tau_k_pus_1  * r_prev[i * (N+1) + j];
                   }

                check_stop[i * (N+1) + j] = w[i * (N+1) + j] - check_stop[i * (N+1) + j];

            }
        }
       // cout<<norma(check_stop)<<endl;
        if (norma(check_stop) < EPS)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

public:
    double *  w;
    double *  final_result;

    NumMethod(int inM, int inN)
    {
        M = inM;
        N = inN;
        h1 = 1.0 * (A2 - A1) / M;
        h2 = 1.0 *  (B2 - B1) / N;

        B = new double [(M+1)*(N+1)];
        w = new double [(M+1)*(N+1)];
        final_result = new double [(M+1)*(N+1)];

        x = new double [M+1];
		y= new double [N+1];

        for (int i = 0; i <= M; i++)
        {
            x[i] = A1 + i * h1;
        }
        for (int j = 0; j <= N; j++)
        {
            y[j] = B1 + j * h2;
        }

        for(int i = 0;i<=M;i++ ){
          for(int j = 0;j<=N;j++ ){
            w[i * (N + 1)+j] = 0;
            final_result[i * (N + 1) + j] = 0;
            }

        }
        for(int i = 0;i<=M;i++ ){
            w[i * (N + 1)] = u_fun(x[i], y[0]);
        }

    }

    double norm_err()
    {
        double ans = 0;
        for (int i = begin_i; i <= min(M, end_i); i++)
        {
            for (int j = begin_j; j <= min(N, end_j); j++)
            {
                ans =  max(abs(w[i * (N+1) + j] - u_fun(x[i],y[j])), ans);
            }

        }
        double all_max_ans;
        MPI_Allreduce(&ans, &all_max_ans, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        return all_max_ans;

    }

    void output()
    {
        if (rank == 0)
            cout << "COMB. RESULT" << endl;
        for (int i = 0; i <= M; i++)
        {
            for (int j = 0; j <= N; j++)
            {
                MPI_Allreduce(&w[i * (N+1) + j], &final_result[i * (N+1) + j], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            }

        }
        if (rank == 0)
        {
            char path[50];
            sprintf(path, "%s%d%s%d%s", "Aproximation_", M,"_",N, ".txt");
            cout << "RESULT 'Aproximation_" << M <<"_"<<N<< ".txt'" << endl;
            ofstream info(path);
            for (int i = 0; i <= M; i++)
            {
                for (int j = 0; j <= N; j++)
                    info << final_result[i * (N+1) + j] << " ";
                info << endl;
            }
            info <<"-----------------"<<endl;

            info.close();
        }
    }

    void Compute_task()
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);// число процессов

         if (rank == 0)
            {
                cout << "hx = " << h1 << "; hy = " << h2 << " EPS: " <<EPS << "  SIZE: " << size <<endl;
            }
            if(size<=3 && size >=1 || size == 5){
                size_i = size;
            }
            else{
                if ( int(log(size)/log(2)) == log(size)/log(2) ){
                    if (size <= 32) {
                        size_i = int(log(size) / log(4)) * 2;
                        if (size == 4) {
                            size_i = 2;
                        }
                    }
                    else {

                        size_i = int(log(size)/log(2)) + 2*d(size,64) -3*d(size,128)+6*d(size,256);
                        size_i-=  9*d(size,512)+8*d(size,1024)-23*d(size,2048);
                    }
                }
          }
        size_j = size / size_i;

        int part_len_i = ceil(1.0 * M / size_i);
        int part_len_j = ceil(1.0 * N / size_j);
        int temp = rank / size_j;
        int temp2 = rank % size_j;
        if(temp == 0)
        {
            begin_i = temp * part_len_i;
        }
        if (temp2 == 0)
        {
            begin_j = (rank % size_j) * part_len_j;
        }
        if (temp != 0)
        {
            begin_i =  temp * part_len_i + 1;
        }
        if (temp2 != 0)
        {
            begin_j = (rank % size_j) * part_len_j + 1;
        }
        end_i = min((temp + 1 ) * part_len_i, M);
        end_j = min(((rank % size_j) + 1 ) * part_len_j, N);
        sv_n =(end_j-begin_j) + 1;
        sv_m =(end_i-begin_i) + 1;
        recieve_top = new double[sv_n];
        recieve_bottom = new double[sv_n];
        send_top = new double [sv_n];
        send_bottom = new double [sv_n];
        recieve_left = new double[sv_m];
        recieve_right = new double[sv_m];
        send_left = new double[sv_m];
        send_right = new double [sv_m];

        B_fun();


        double start = MPI_Wtime();
        int i = 0;
        while (Iteration(EPS) == 0)
        {


            ++i;
        }
        double finish = MPI_Wtime();


        double full_norm_err = norm_err();
        if (rank == 0)
        {
            cout << "--------RESULT---------------------" << endl;
            cout << "ITERATIONS: " << i << endl;
            cout << "ERROR: " << full_norm_err << endl;
            cout << "TIME: " << finish - start << " s" << endl;
            cout << "-----------------------------------" << endl;
        }

        output();
    }

};


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(argc != 3)
    {
        cout << "Input looks like: ./prog M N " << endl;
        MPI_Finalize();
        return 0;
    }


    NumMethod Aproximation(atoi(argv[1]),atoi(argv[2]));
    cout<<"----------BEGIN-----------------"<<endl;
    Aproximation.Compute_task();

    MPI_Finalize();
    return 0;
}
