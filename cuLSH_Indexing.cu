/*
	Copyright (C) 2014 Aris Kagias <ariskagias@gmail.com>. All Rights Reserved.
	
	This file is part of cuLSH.
	cuLSH is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.
	
	You should have received a copy of the GNU General Public License
	along with cuLSH.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef __cuLSH__Indexing__
#define __cuLSH__Indexing__

namespace cuLSH {
//###########################################################################################################
bool projectMatrix(
	float* d_projectedMatrix,	// result matrix, must have size [N x M]
	const float* d_matrix,	// input matrix, [D x N]
	const float* d_A,	// A matrix, must have size [D x M]
	const float* d_b,	// b vector, must have length M
	const float W,	// bucket width
	const int N, const int D, const int M	// # of data vectors, # of original dimensions, # of projection dimensions
	)
{
	const float alpha = 1.0, beta = 0.0;
	cublasHandle_t handle;
	cublasStatus_t status;
	if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS) {
		DEF_PRINT_ERROR("Cublas initialization failed...");
		return false;
		}
	status = cublasSgemm(
		handle,
		CUBLAS_OP_T, CUBLAS_OP_N,
		N, M, D,
		&alpha,
		d_matrix, D,
		d_A, D,
		&beta,
		d_projectedMatrix, N);
//	cudaDeviceSynchronize();
	
	if(status != CUBLAS_STATUS_SUCCESS) {
		DEF_PRINT_ERROR("Matrix multiplication failed (cublasStatus = %d)...\n", status);
		return false;
		}
		
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (N + BLOCK_SIZE - 1)/BLOCK_SIZE, (M + BLOCK_SIZE - 1)/BLOCK_SIZE );
	kernel_subtractVectorFromMatrixRows <<< dimGrid, dimBlock >>> (d_projectedMatrix, d_b, N, M);
	kernel_divideMatrixWithScalar <<< dimGrid, dimBlock >>> (d_projectedMatrix, W, N, M);
//	cudaDeviceSynchronize();
	
	return true;
}
//###########################################################################################################
void radixSortRows_1matrix(
	ThrustUnsignedD& d_permutation,
	const ThrustFloatD& d_matrix,
	const int rows, const int columns,
	FILE* debugStream = 0
	)
{
	const char *funcString = "[radixSortRows_1matrix]";
	
	ThrustFloatD d_column(rows);
	
	d_permutation.resize(rows);
	thrust::sequence(d_permutation.begin(), d_permutation.end());
	
	if(debugStream) fprintf(debugStream, "%s\tRadix sorting matrix rows. Columns examined(%d..1): ", funcString, columns);
	for(int col=columns-1; col>=0; col--) {
		if(debugStream) fprintf(debugStream, "%d ", col+1);
		thrust::gather(d_permutation.begin(), d_permutation.end(), d_matrix.begin() + col*rows, d_column.begin());
		thrust::stable_sort_by_key(d_column.begin(), d_column.end(), d_permutation.begin());
		}
	if(debugStream) fprintf(debugStream, "\n");
//	cudaDeviceSynchronize();
}
//###########################################################################################################
struct findUniqueMatrixRows_isOne {
	__host__ __device__ bool operator()(const unsigned char x) {
		return (bool) x;
		}
	};
struct findUniqueMatrixRows_floatORchar {
	__host__ __device__ unsigned char operator()(float num1, unsigned char num2) {
		return (num1==0) ? num2 : 1;
		}
	};
void calculateBuckets(
	ThrustUnsignedD& d_buckets_indices,
	ThrustUnsignedD& d_buckets_sizes,
	const ThrustFloatD& d_projectedMatrix,
	const int N, const int M,
	const ThrustUnsignedD& d_permutation
	)
{
	ThrustFloatD d_column(N);
	ThrustFloatD d_diff_temp(N);
	thrust::device_vector<unsigned char> d_diff(N, 0);
		
	// Calculate each row's difference from above row
//	if(debugStream) fprintf(debugStream, "%s\tChecking for unique rows...\n", funcString);
	for(int col=0; col<M; col++) {
		thrust::gather(d_permutation.begin(), d_permutation.end(), d_projectedMatrix.begin() + col*N, d_column.begin());
		thrust::adjacent_difference(d_column.begin(), d_column.end(), d_diff_temp.begin());
		thrust::transform(d_diff_temp.begin(), d_diff_temp.end(), d_diff.begin(), d_diff.begin(), findUniqueMatrixRows_floatORchar());
		}
	
	// Place 1 in first element of d_diff (in case first row of d_matrix is a zero vector)
	d_diff[0] = 1;
	
	// Extract unique row indices (indices refer to permutation vector d_permutation)
	int buckets = thrust::count(d_diff.begin(), d_diff.end(), 1);
	
	d_buckets_indices.resize(buckets);
	thrust::copy_if(thrust::make_counting_iterator(0), thrust::make_counting_iterator(N), d_diff.begin(), d_buckets_indices.begin(), findUniqueMatrixRows_isOne());
	
	d_buckets_sizes.resize(buckets);
	thrust::adjacent_difference(d_buckets_indices.begin() + 1, d_buckets_indices.end(), d_buckets_sizes.begin());
	d_buckets_sizes[buckets - 1] = N - d_buckets_indices[buckets - 1];
	
//	cudaDeviceSynchronize();
//	printf("Unique rows found: %d\n", buckets);

}
//###########################################################################################################
void copyMatrixRows(
	float* d_destination,
	const unsigned destinationRows,
	const unsigned columns,
	const float* d_source,
	const unsigned sourceRows,
	const unsigned* d_rows2copy_indices
	)
{	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid( (destinationRows + BLOCK_SIZE - 1) / BLOCK_SIZE, (columns + BLOCK_SIZE - 1) / BLOCK_SIZE );
	kernel_copyMatrixRows <<< dimGrid, dimBlock >>> (
		d_destination,
		destinationRows,
		columns,
		d_source,
		sourceRows,
		d_rows2copy_indices
		);
//	cudaDeviceSynchronize();
}
//###########################################################################################################
}	// end of namespace

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

namespace cuLSH {

// index only one table
// only reason for double pointers is because we are allocating memory for them in this function
bool indexTableData(
	unsigned* buckets,
	unsigned** buckets_totalIndices,
	unsigned** buckets_startingIndices,
	unsigned** buckets_sizes,
	float** buckets_codes,
	const float* matrix,
	const float* A,
	const float* b,
	const float W,
	const int N, const int D, const int M,
	FILE* debugStream = 0
	)
{
	const char *funcString = "[indexTableData]";
	
	ThrustFloatD d_matrix(matrix, matrix + N * D);
	ThrustFloatD d_projectedMatrix(N * M);
	ThrustFloatD d_A(A, A + D * M);
	ThrustFloatD d_b(b, b + M);
	
	if(debugStream) fprintf(debugStream, "%s\tProjecting and flooring data matrix...\n", funcString);
	
	if(!projectMatrix(
			thrust::raw_pointer_cast(d_projectedMatrix.data()),
			thrust::raw_pointer_cast(d_matrix.data()),
			thrust::raw_pointer_cast(d_A.data()),
			thrust::raw_pointer_cast(d_b.data()),
			W,
			N, D, M
			)) return false;
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((N + BLOCK_SIZE - 1)/BLOCK_SIZE, (M + BLOCK_SIZE - 1)/BLOCK_SIZE);
	kernel_floorMatrix <<< dimGrid, dimBlock >>> (thrust::raw_pointer_cast(d_projectedMatrix.data()), N, M);
	// No need to place cudaDeviceSynchronize here. Next time projectedMatrix is needed is in the next device call, on the same stream.
	
	// Free device memory
	d_matrix.clear(); d_matrix.shrink_to_fit();
	d_A.clear(); d_A.shrink_to_fit();
	d_b.clear(); d_b.shrink_to_fit();
	
	// Create bucket objects on device
	ThrustUnsignedD d_sortedRows_indices(N);
	ThrustUnsignedD d_buckets_indices(0);	// will be resized in calculateBuckets
	ThrustUnsignedD d_buckets_sizes(0);	// will be resized in calculateBuckets

	// Radix sort projected matrix rows
	radixSortRows_1matrix(d_sortedRows_indices, d_projectedMatrix, N, M, debugStream);
	
	if(debugStream) fprintf(debugStream, "%s\tCalculating number of buckets, starting indices and sizes...", funcString);
	calculateBuckets(d_buckets_indices, d_buckets_sizes, d_projectedMatrix, N, M, d_sortedRows_indices);
	*buckets = d_buckets_indices.size();
	
	if(debugStream) fprintf(debugStream, " -----> %d distinct buckets\n", *buckets);
	
	// Extract unique codes
	ThrustFloatD d_buckets_codes(*buckets * M);
	ThrustUnsignedD d_buckets_globalIndices(*buckets);
	thrust::gather(d_buckets_indices.begin(), d_buckets_indices.end(), d_sortedRows_indices.begin(), d_buckets_globalIndices.begin());
	copyMatrixRows(
		thrust::raw_pointer_cast(d_buckets_codes.data()),
		*buckets, M,
		thrust::raw_pointer_cast(d_projectedMatrix.data()),
		N,
		thrust::raw_pointer_cast(d_buckets_globalIndices.data())
		);
	
	// Allocate memory on host for bucket objects
	if(!(*buckets_totalIndices = (unsigned*) malloc(N * sizeof(unsigned)))) {
		DEF_PRINT_ERROR("Could not allocate memory for bucket total indices on host...\n");
		return false;
		}
	if(!(*buckets_startingIndices = (unsigned*) malloc(*buckets * sizeof(unsigned)))) {
		DEF_PRINT_ERROR("Could not allocate memory for buckets starting indices on host...\n");
		return false;
		}
	if(!(*buckets_sizes = (unsigned*) malloc(*buckets * sizeof(unsigned)))) {
		DEF_PRINT_ERROR("Could not allocate memory for bucket sizes on host...\n");
		return false;
		}
	if(!(*buckets_codes = (float*) malloc(*buckets * M * sizeof(float)))) {
		DEF_PRINT_ERROR("Could not allocate memory for bucket codes on host...\n");
		return false;
		}
	
	// Copy bucket objects to host
	thrust::copy(d_sortedRows_indices.begin(), d_sortedRows_indices.end(), *buckets_totalIndices);
	thrust::copy(d_buckets_indices.begin(), d_buckets_indices.end(), *buckets_startingIndices);
	thrust::copy(d_buckets_sizes.begin(), d_buckets_sizes.end(), *buckets_sizes);
	thrust::copy(d_buckets_codes.begin(), d_buckets_codes.end(), *buckets_codes);
	
	return true;
}

}	// end of namespace

#endif	// #ifndef __cuLSH__Indexing__
