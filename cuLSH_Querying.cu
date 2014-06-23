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

#ifndef __cuLSH__Querying__
#define __cuLSH__Querying__
#include "cuLSH_Indexing.cu"

namespace cuLSH {
//###########################################################################################################
/*
	Radix sort 2 matrices, concatenated by placing the 2nd matrix under the 1st
	The permutation vector pointing to the matrices' rows is returned
	Indices [0 .. rows1) of the permutation vector refer to the first matrix's rows, indices [rows1 .. rows1 + rows2) to the second matrix's rows
*/
void radixSortRows_2matrices(
	ThrustUnsignedD& d_permutation,
	const ThrustFloatD& d_matrix1,
	const ThrustFloatD& d_matrix2,
	const int rows1, const int rows2,
	const int columns,
	FILE* debugStream = 0
	)
{
	const char *funcString = "[radixSortRows_2matrices]";
	ThrustFloatD d_column(rows1+rows2);
	ThrustFloatD d_columnGathered(rows1+rows2);
	
	d_permutation.resize(rows1+rows2);
	thrust::sequence(d_permutation.begin(), d_permutation.end());
	
	if(debugStream) fprintf(debugStream, "%s\tRadix sorting matrices' rows. Columns examined(%d..1): ", funcString, columns);
	for(int col = columns-1; col>=0; col--) {
		if(debugStream) fprintf(debugStream, "%d ", col + 1);
		thrust::copy_n(d_matrix1.begin() + col*rows1, rows1, d_column.begin());
		thrust::copy_n(d_matrix2.begin() + col*rows2, rows2, d_column.begin() + rows1);
		thrust::gather(d_permutation.begin(), d_permutation.end(), d_column.begin(), d_columnGathered.begin());
		thrust::stable_sort_by_key(d_columnGathered.begin(), d_columnGathered.end(), d_permutation.begin());
		}
	if(debugStream) fprintf(debugStream, "\n");
//	cudaDeviceSynchronize();
}
//###########################################################################################################
/*
	Generate mulriprobe codes of queries, after projection is done (without flooring the results)
	d_projectedQueries is [Q x M] at first, containing the (non-floored) result of the projection
	In the end, d_projectedQueries is [Q*T x M], containing the (floored) multiprobe codes of the queries
*/
void createMultiprobingCodes(
	ThrustFloatD& d_projectedQueries,
	const int Q,
	const int M,
	const int T
	)
{
	dim3 dimBlock;
	dim3 dimGrid;
	// Configure deltas
	ThrustFloatD d_deltas(Q * 2 * M);	
	dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dimGrid = dim3( (Q + BLOCK_SIZE - 1)/BLOCK_SIZE, (M + BLOCK_SIZE - 1)/BLOCK_SIZE );
	kernel_calculateDeltas <<< dimGrid, dimBlock >>> (
		thrust::raw_pointer_cast(d_projectedQueries.data()),
		Q, M,
		thrust::raw_pointer_cast(d_deltas.data())
		);
	// d_projectedQueries are floored inside kernel_calculateDeltas
	
	// Convert d_projectedQueries to [(Q * T) x M]
	ThrustFloatH h_projectedQueries_nonFloored(d_projectedQueries);	// [Q x M] non floored codes
	ThrustFloatH h_projectedQueries_mp( Q * T * M );
	
	ThrustUnsignedH h_indices(Q, T);
	thrust::exclusive_scan(h_indices.begin(), h_indices.end(), h_indices.begin());
	// Now h_indices contain [0, T, 2*T, ... (Q-1)*T], element <i> represents row with authentic code of query <i>
	// Copy authentic (non-floored) codes of queries to proper rows of h_projectedQueries_mp
	for(int column = 0; column < M; column++)
		thrust::scatter(
			h_projectedQueries_nonFloored.begin() + column * Q,
			h_projectedQueries_nonFloored.begin() + (column+1) * Q,
			h_indices.begin(),
			h_projectedQueries_mp.begin() + column * Q * T
			);
	
	// Resize projected query codes matrix from [Q x M] to [(Q * T) x M]
	d_projectedQueries.resize(Q * T * M);
	thrust::copy(h_projectedQueries_mp.begin(), h_projectedQueries_mp.end(), d_projectedQueries.begin());
	
	// Calculate multiprobing codes
	ThrustUint64D d_combinations(Q * T);
	ThrustFloatD d_combinations_deltas(Q * T);
	
	dimBlock = dim3(BLOCK_SIZE * BLOCK_SIZE, 1);
	dimGrid = dim3( (Q + BLOCK_SIZE * BLOCK_SIZE - 1)/(BLOCK_SIZE * BLOCK_SIZE), 1 );
	
	kernel_calculateProbingCodes <<< dimGrid, dimBlock >>> (
		thrust::raw_pointer_cast(d_projectedQueries.data()),
		Q, M, T,
		thrust::raw_pointer_cast(d_deltas.data()),
		thrust::raw_pointer_cast(d_combinations.data()),
		thrust::raw_pointer_cast(d_combinations_deltas.data())
		);
	
//	cudaDeviceSynchronize();
}
//###########################################################################################################
/*
	Find the buckets matching queries' codes
*/
void findMatchingBuckets(
	int* queryBuckets,	// [T x Q]
	const ThrustFloatD& d_projectedQueries,	// [Q x M]
	const ThrustFloatD& d_bucketCodes,	// [B x M]
	const int Q, const int B,
	const int M, const int T,
	FILE* debugStream = 0
	) 
{
	
	ThrustUnsignedD d_perm(B+Q*T);
	radixSortRows_2matrices(d_perm, d_bucketCodes, d_projectedQueries, B, Q*T, M, debugStream);
	
	ThrustUnsignedH h_perm(d_perm);
	
	ThrustIntH h_tableBuckets(Q*T , -1);	// concatenated Q vectors of T elements (T, 2T, ... QT)
	
	unsigned queryIndex;
	unsigned probeIndex;
	
// 	EXPLAIN WHY START ITERATING FROM 1
	for(int i=1; i<B+Q*T; i++) if(h_perm[i]>=B) {
		queryIndex = (h_perm[i] - B)/T;
		probeIndex = (h_perm[i] - B)%T;
		if( (h_tableBuckets[ queryIndex * T + probeIndex ] = h_perm[i-1]) >= B )
			h_tableBuckets[ queryIndex * T + probeIndex ] = h_tableBuckets[ ((h_perm[i-1] - B)/T) * T + (h_perm[i-1] - B)%T ];
		}
	
	ThrustIntD d_queryBuckets(h_tableBuckets);
	
	dim3 dimBlock(1, BLOCK_SIZE);
	dim3 dimGrid(1, (Q*T + BLOCK_SIZE - 1)/BLOCK_SIZE);
	kernel_findMatchingBuckets_evaluate <<< dimGrid, dimBlock >>> (
		thrust::raw_pointer_cast(d_queryBuckets.data()),
		thrust::raw_pointer_cast(d_projectedQueries.data()),
		thrust::raw_pointer_cast(d_bucketCodes.data()),
		Q*T, M, B
		);
//	cudaDeviceSynchronize();
	
	thrust::copy(d_queryBuckets.begin(), d_queryBuckets.end(), queryBuckets);
}
//###########################################################################################################
/*
	Find the buckets matching to queries
	Firstly, the multiprobe codes are generated and then they are matched to the buckets
*/
bool findTableQueryBins(
	int* queryBuckets,
	const float* queries,
	const float* A,
	const float* b,
	const float W,
	const int Q, const int D, const int M, const int T,
	const float* buckets_codes,
	const unsigned B,
	FILE* debugStream = 0
	)
{
	const char *funcString = "[findTableQueryBins]";
	ThrustFloatD d_queries(queries, queries + Q * D);
	ThrustFloatD d_projectedQueries(Q * M);
	ThrustFloatD d_A(A, A + D * M);
	ThrustFloatD d_b(b, b + M);
	
	if(debugStream) fprintf(debugStream, "%s\tProjecting query matrix...\n", funcString);
	
	if(!projectMatrix(
			thrust::raw_pointer_cast(d_projectedQueries.data()),
			thrust::raw_pointer_cast(d_queries.data()),
			thrust::raw_pointer_cast(d_A.data()),
			thrust::raw_pointer_cast(d_b.data()),
			W,
			Q, D, M)) return false;
	// d_projectedQueries are non-floored
	
	// Free device memory
	d_queries.clear(); d_queries.shrink_to_fit();
	d_A.clear(); d_A.shrink_to_fit();
	d_b.clear(); d_b.shrink_to_fit();
	
	// Load bucket codes to device memory
	ThrustFloatD d_buckets_codes(buckets_codes, buckets_codes + B * M);
	
	if(debugStream) fprintf(debugStream, "%s\tCreating multiprobing codes...\n", funcString);
	
	// Generate mulriprobing codes for queries
if(T > 1)
	createMultiprobingCodes(d_projectedQueries, Q, M, T);
else {
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((Q + BLOCK_SIZE - 1)/BLOCK_SIZE, (M + BLOCK_SIZE - 1)/BLOCK_SIZE);
	kernel_floorMatrix <<< dimGrid, dimBlock >>> (thrust::raw_pointer_cast(d_projectedQueries.data()), Q, M);
	}
	// d_projectedQueries matrix has now size [(Q * T] x M], and is floored
	
	if(debugStream) fprintf(debugStream, "%s\tFinding matching bucket codes...\n", funcString);
	
	findMatchingBuckets(queryBuckets, d_projectedQueries, d_buckets_codes, Q, B, M, T, debugStream);
	
	return true;
}
//###########################################################################################################
//####################################################################################################
/*
	Extract K nearest neighbors, after the K smallest distances have been extracted
*/
void calculateIds(
	ThrustIntD& d_knnIds,
	ThrustFloatD& d_distances,
	const int Q, const int K,
	const ThrustUnsignedD& d_queryCandidates_totalIndices,
	const ThrustUnsignedD& d_queryCandidates_startingIndices,
	const ThrustUnsignedD& d_queryCandidates_sizes
	)
{
	// Define total number of candidates for all queries
	const unsigned totalIndices = d_queryCandidates_totalIndices.size();
	// Initialize heap for storing top K smallest distances for each query
	ThrustFloatD d_heap(Q * K, FLT_MAX);
	d_knnIds.resize(Q * K);
	// Initialize all knn indices to -1 (no candidate assigned)
	thrust::fill(d_knnIds.begin(), d_knnIds.end(), -1);
	
	dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE, 1);
	dim3 dimGrid( (Q + BLOCK_SIZE * BLOCK_SIZE - 1) / (BLOCK_SIZE * BLOCK_SIZE), 1 );
	
	
	// Calculate knn indices with smallest distances for each query
	kernel_configureIds <<< dimGrid, dimBlock >>> (
		thrust::raw_pointer_cast(d_heap.data()),
		thrust::raw_pointer_cast(d_knnIds.data()),
		Q, K, totalIndices,
		thrust::raw_pointer_cast(d_distances.data()),
		thrust::raw_pointer_cast(d_queryCandidates_totalIndices.data()),
		thrust::raw_pointer_cast(d_queryCandidates_startingIndices.data()),
		thrust::raw_pointer_cast(d_queryCandidates_sizes.data())
		);
	
	// Sort calculated knn indices and corresponding distances in heap according to their distance for each query
	kernel_sortIds <<< dimGrid, dimBlock >>> (
		thrust::raw_pointer_cast(d_heap.data()),
		thrust::raw_pointer_cast(d_knnIds.data()),
		Q,
		K
		);
	cudaDeviceSynchronize();
	// Resize distances to K * Q, and assign the distance heap to it
//	d_distances.resize(K * Q); d_distances.shrink_to_fit();
//	printf("Size of heap: %d\tSize of distances: %d\n", d_heap.size(), d_distances.size());
	d_distances.resize(K * Q);
	//thrust::copy_n(d_heap.begin(), DEF_MIN(K * Q, totalIndices), d_distances.begin());
	thrust::copy(d_heap.begin(), d_heap.end(), d_distances.begin());
}

//#####################################################################################################
//####################################################################################################
/*
	Calculate distances between queries and their candidates
*/
void calculateDistances(
	ThrustFloatD& d_distances,
	const ThrustFloatD& d_queries,
	const int D,
	const int Q,
	const ThrustFloatD& d_dataset,
	const int N,
	const ThrustUnsignedD& d_candidateIndices,
	const ThrustUnsignedD& d_queryIndices,
	FILE* debugStream = 0
	)
{
	const char *funcString = "[CalculateDistances]";
	float time_ms;
	float totaltime_ms = 0.0;
	cudaEvent_t ev1, ev2;
	cudaEventCreate(&ev1);
	cudaEventCreate(&ev2);
	
	const unsigned total = d_candidateIndices.size();
	d_distances.resize(total);
	
	unsigned candidatesPerBlock = DEF_MIN(total, 65535 * BLOCK_SIZE_Y);
	unsigned totalBlocks = (total + candidatesPerBlock -1) / candidatesPerBlock;
	dim3 dimBlock(BLOCK_SIZE_REDUCE, BLOCK_SIZE_Y);
	dim3 dimGrid(1, (candidatesPerBlock + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
	unsigned start;
	
	for(int block = 0; block < totalBlocks; block++) {
		start = block * candidatesPerBlock;
		if(block == totalBlocks - 1) {
			candidatesPerBlock = total - start;
			dimGrid = dim3(1, (candidatesPerBlock + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
			}
		
		cudaEventRecord(ev1);
	
		kernel_calculateDistances <<< dimGrid, dimBlock >>> (
			thrust::raw_pointer_cast(d_distances.data()) + start,
			candidatesPerBlock,
			thrust::raw_pointer_cast(d_queries.data()),
			D,
			Q,
			thrust::raw_pointer_cast(d_dataset.data()),
			N,
			thrust::raw_pointer_cast(d_candidateIndices.data()) + start,
			thrust::raw_pointer_cast(d_queryIndices.data()) + start
			);
		
		cudaEventRecord(ev2);
		cudaEventSynchronize(ev2);
		cudaEventElapsedTime(&time_ms, ev1, ev2);
		totaltime_ms += time_ms;
		
		if(debugStream) fprintf(debugStream, "%s: Block %d/%d - %u candidates, TIME: %fms. -> %.2fMF\n", funcString, block + 1, totalBlocks, candidatesPerBlock, time_ms, ((candidatesPerBlock * 128 * 3) / (time_ms/1000.0))/1000000.0 );
		}
	
	if(debugStream) fprintf(debugStream, "KNN review: %u candidates, %f milliseconds, %f seconds, %.3fMF\n", total, totaltime_ms, totaltime_ms/1000.0, ((total / totaltime_ms) * (D * 3 * 1000.0))/ 1000000.0);

}
//####################################################################################################

}	// end of namespace
#endif	// #ifndef __cuLSH__Querying__

