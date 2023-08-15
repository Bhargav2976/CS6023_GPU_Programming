#include<iostream>
#include<sys/time.h>
#include<cuda.h>
# define TILE_DIM 32
using namespace std;

__global__ void sharedABMultiply(int *a, int* b, int *e,int p,int q,int r){
	__shared__ int aTile[TILE_DIM][TILE_DIM];
	__shared__ int bTile[TILE_DIM][TILE_DIM];
	unsigned row1 = blockIdx.y*blockDim.y + threadIdx.y; //row for A
	unsigned column2  = blockIdx.x*blockDim.x + threadIdx.x; //column for B
	float sum = 0.0f;
	int no_of_tiles = ceil(float(q)/TILE_DIM);
	for (int i = 0;i<no_of_tiles;i++){
		int column1 = i*blockDim.x + threadIdx.x;//column for A
		int row2 = i*blockDim.y + threadIdx.y;//row for B
		if(row1<p && column1<q){
			aTile[threadIdx.y][threadIdx.x] = a[row1*q + column1];
		}
		else{
			aTile[threadIdx.y][threadIdx.x] = 0;
		}
		if(row2<q && column2<r){
			bTile[threadIdx.y][threadIdx.x] = b[row2*r + column2];     
		}
		else{
			bTile[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();
		for(int j = 0;j<TILE_DIM;j++){
			sum += aTile[threadIdx.y][j]*bTile[j][threadIdx.x];
		}
		__syncthreads();
	}
	if(row1<p && column2<r){
		e[row1*r + column2] = sum;
	}
}
__global__ void transposeCoalesced(int *odata, int *idata,int p, int q, int r)
{
  __shared__ int tile[TILE_DIM][TILE_DIM];

  int column = blockIdx.x * TILE_DIM + threadIdx.x; //col
  int row = blockIdx.y * TILE_DIM + threadIdx.y; //row

  if(column<q && row<r){
	tile[threadIdx.y][threadIdx.x] = idata[(row)*q + column];
  }
  __syncthreads();

  column = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  row = blockIdx.x * TILE_DIM + threadIdx.y;
  if(column < r && row<q){
	odata[(row)*r + column] = tile[threadIdx.x][threadIdx.y];
  }
  
}

__global__ void sharedCDTMultiply(int *c, int* d, int *f,int p,int q,int r)
{
	__shared__ int cTile[TILE_DIM][TILE_DIM];
	__shared__ int dTile[TILE_DIM][33];
	int row1 = blockIdx.y*blockDim.y + threadIdx.y; //row for C
	int row2  = blockIdx.x*blockDim.x + threadIdx.y; //row for D
	int column = blockIdx.x*blockDim.x + threadIdx.x; // column for E
	float sum = 0.0f;
	int no_of_tiles = ceil(float(q)/TILE_DIM);
	for (int i = 0;i<no_of_tiles;i++){
		int column1 = i*blockDim.x + threadIdx.x;//column for C
		int column2 = i*blockDim.y + threadIdx.x;//column for D
		if(row1<p && column1 < q){
			cTile[threadIdx.y][threadIdx.x] = c[row1*q + column1];
		}
		else{
			cTile[threadIdx.y][threadIdx.x] = 0;
		}
		if(row2 < r && column2 < q){
            dTile[threadIdx.y][threadIdx.x] = d[row2*r + column2];
		}
		else{
			dTile[threadIdx.y][threadIdx.x] = 0;
		}
		__syncthreads();
	    for(int j = 0;j<TILE_DIM;j++){
			sum += cTile[threadIdx.y][j]*dTile[threadIdx.x][j];
		}
	    __syncthreads();
	}
	
	if(row1<p && column <r){
		f[row1*r + column] = sum;
	}
	
}
__global__ void sharedmatrixaddn(int *e,int *f,int *g,int p,int q, int r){
	__shared__ int eTile[TILE_DIM][TILE_DIM];
	__shared__ int fTile[TILE_DIM][TILE_DIM];
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int column = blockIdx.x*blockDim.x + threadIdx.x;
	if(row<p&& column<r){
		eTile[threadIdx.y][threadIdx.x] = e[row*r + column];
	    fTile[threadIdx.y][threadIdx.x] = f[row*r + column];
	}
	else{
		eTile[threadIdx.y][threadIdx.x] = 0;
	    fTile[threadIdx.y][threadIdx.x] = 0;
	}
	__syncthreads();
	if(row < p && column<r){
		g[row*r + column] = eTile[threadIdx.y][threadIdx.x] + fTile[threadIdx.y][threadIdx.x];
	}
}


// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE, *d_matrix1, *d_matrix2,*d_matrixDT;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));
    
	cudaMalloc(&d_matrix1,p*r*sizeof(int));
	cudaMalloc(&d_matrix2,p*r*sizeof(int));
	cudaMalloc(&d_matrixDT,q*r*sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	//kernel-1
	dim3 grid1(ceil(float(r)/TILE_DIM),ceil(float(p)/TILE_DIM),1);
	dim3 block1(TILE_DIM,TILE_DIM,1);
	sharedABMultiply<<<grid1,block1>>>(d_matrixA,d_matrixB,d_matrix1,p,q,r);

	dim3 gridT(ceil(float(q)/TILE_DIM),ceil(float(r)/TILE_DIM),1);
	dim3 blockT(TILE_DIM,TILE_DIM,1);
	transposeCoalesced<<<gridT,blockT>>>(d_matrixDT,d_matrixD,p,q,r);
	 
	//kernel - 2
	dim3 grid2(ceil(float(r)/TILE_DIM),ceil(float(p)/TILE_DIM),1);
	dim3 block2(TILE_DIM,TILE_DIM,1);
	sharedABMultiply<<<grid2,block2>>>(d_matrixC,d_matrixDT,d_matrix2,p,q,r);
    
	//kernel -3
    dim3 grid3(ceil(float(r)/TILE_DIM),ceil(float(p)/TILE_DIM),1);
	dim3 block3(TILE_DIM,TILE_DIM,1);
	sharedmatrixaddn<<<grid3,block3>>>(d_matrix1,d_matrix2,d_matrixE,p,q,r);
	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
	cudaFree(d_matrix1);
	cudaFree(d_matrix2);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	
