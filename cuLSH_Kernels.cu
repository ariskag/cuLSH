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

#ifndef __cuLSH__Kernels__
#define __cuLSH__Kernels__

namespace cuLSH{
//###########################################################################################################
/*
	Exchange elements at two given positions of a vector
	Used for balancing heaps of deltas, and heaps of distances(& knn_ids)
*/
template <typename T>
__device__ void exchange(T* vector, const int& pos1, const int& pos2)
{
	T temp = vector[pos1];
	vector[pos1] = vector[pos2];
	vector[pos2] = temp;
}
//###########################################################################################################
/*
	Subtract vector from each row of a matrix
	Used to perform the subtraction AA - b, where AA = X' * A
*/
__global__ void kernel_subtractVectorFromMatrixRows(
	float* matrix,	// matrix of size [rows x columns]
	const float* vector,	// vector of length [columns]
	const int rows, const int columns
	)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int column = blockIdx.y * blockDim.y + threadIdx.y;
	if(row < rows && column < columns) matrix[column * rows + row] -= vector[column];
}

/*
	Divide each element of a matrix with a scalar value
	Used to perform the division AB / W, where AB = bsxfun(@minus, (X' * A), b)
*/
__global__ void kernel_divideMatrixWithScalar(
	float* matrix,	// matrix of size [rows x columns]
	const float scalar,
	const int rows, const int columns
	)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int column = blockIdx.y * blockDim.y + threadIdx.y;
	if(row < rows && column < columns) matrix[column * rows + row] /= scalar;
}
//###########################################################################################################
/*
	Copy desired rows of a source matrix to a destination matrix
	Used to copy unique codes of projected dataset to buckets' code matrix
	Source and destination matrix must have same number of columns
*/
__global__ void kernel_copyMatrixRows(
	float* destination,
	const unsigned destinationRows,
	const unsigned columns,
	const float* source,
	const unsigned sourceRows,
	const unsigned* rows2copy_indices
	)
{
	unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
	// x: row index
	// y: column index
	if(x < destinationRows && y < columns) destination[y * destinationRows + x] = source[y * sourceRows + rows2copy_indices[x]];
}
//###########################################################################################################
/*
	Floor each element of a matrix
	Used to perform floor(ABC), where ABC = bsxfun(@minus, (X' * A), b) / W
*/
__global__ void kernel_floorMatrix(
	float* matrix,
	const unsigned rows,
	const unsigned columns
	)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int column = blockIdx.y * blockDim.y + threadIdx.y;
	
	if(row < rows && column < columns) matrix[column * rows + row] = floorf(matrix[column * rows + row]);
}
//###########################################################################################################
/*
	Perform square root calculation for each element of a matrix
	Used to calculate real values of euclidean distances when nearest neighbors have been found at the end of SearchTables::search()
	Can by bypassed if the user does not need the distances. In this case, the squares of the distances will be returned
*/
__global__ void kernel_squareRootMatrix(
	float* matrix,
	const unsigned rows,
	const unsigned columns
	)
{
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int column = blockIdx.y * blockDim.y + threadIdx.y;
	if(row < rows && column < columns) matrix[column * rows + row] = sqrtf(matrix[column * rows + row]);
}
//###########################################################################################################
/*
	Decide whether given buckets for each query are correct or not
	Called after approximating queries' matching buckets at findMatchingBuckets()
	If a bucket is not valid for a query, -1 is assigned at its position in queries' bucket indices matrix
*/
__global__ void kernel_findMatchingBuckets_evaluate(
	int* queryBuckets,	// [T x Q]
	const float* queryCodes,	// [(QT = Q * T] x M]
	const float* bucketCodes,	// [B x M]
	const unsigned QT, const unsigned M,
	const unsigned B
	)
{
	unsigned index = blockIdx.y * blockDim.y + threadIdx.y;
	int i;
	if(index < QT && queryBuckets[index] >= 0) {
		for(i = 0; i < M; i++) if(queryCodes[i * QT + index] != bucketCodes[i * B + queryBuckets[index]]) break;
		if(i < M) queryBuckets[index] = -1;
		}
}
//###########################################################################################################
/*
	Balance heap of indices to delta values after a new delta value index has been inserted into it
	The heap is initialized with -1 values, so take into account that heap might not be full yet
*/
__device__ void balanceHeap_deltas(
	int* permutation,
	const float* deltas,
	const int& Q,
	const int& query,
	const int& total,
	const int& totalLevels
	)
{
	int parent = 0, left, right, maxIndex;
	for(int level = 0; level < totalLevels - 1; level++) {
		left = parent * 2 + 1;
		right = left + 1;
		maxIndex = parent;
		
		if(left < total && permutation[left] == -1)
			maxIndex = left;
		else if(right < total && permutation[right] == -1)
			maxIndex = right;
		else {
			if(left < total && deltas[permutation[left] * Q + query] > deltas[permutation[maxIndex] * Q + query])
				maxIndex = left;
			if(right < total && deltas[permutation[right] * Q + query] > deltas[permutation[maxIndex] * Q + query])
				maxIndex = right;
			}
		
		if(maxIndex == parent) break;
		
		exchange(permutation, parent, maxIndex);
		parent = maxIndex;
		}
}

/*
	Calculate probing codes of queries
	<deltas> have already been calculated, but are not yet sorted in ascending order
	When sorting <deltas>, the <deltas> sequence is unchanged, only a vector referring to deltas positions is changed.
		This happens because it matters if a delta is in the first M positions of the <deltas> sequence or the last ones.
			Deltas in positions [0, M) refer to subtracting 1 from [0, M) dimensions of original code
			Deltas in positions [M+1, 2*M) refer to adding  1 to [0, M) dimensions of original code
	<combinations> are 64bit integers. Bit <i> of a number is set if delta <i> (in the sorted sequence) is used to create probing code
	<combinations_deltas> contain the sums of the deltas assigned to each <combinations> 64bit integer.
	At each iteration of the probing code generation, the <combinations> element with the smallest sum of deltas is used
*/
__global__ void kernel_calculateProbingCodes(
	float* queries,
	const int Q, const int M, const int T,
	const float* deltas,
	uint64_t* combinations,
	float* combinations_deltas
	)
//	queries: [(Q * T) x M]
//	deltas: [Q x 2M], first M -> DOWN, last M -> UP	(column * Q + query)
//	combinations, combinations_deltas: [Q x T]		(t * Q + query)
{
	int query = blockIdx.x * blockDim.x + threadIdx.x;
	if(query < Q) {
		// Make a permutation array referring to sorted delta elements
		int totalLevels = __float2int_ru( log2f( (float) (2 * M + 1) ) );
		int maxT = DEF_MIN(2 * M, 64);	// Keep at most the 64 smallest deltas (in case 2*M>64)
		
		int perm[64];	// permutation vector, after sorting it contains indices to deltas in ascending order
		for(int i = 0; i < maxT; i++) perm[i] = -1;
		for(int i = 0; i < 2 * M; i++) if(perm[0] == -1 || deltas[i * Q + query] < deltas[perm[0] * Q + query]) {
			perm[0] = i;
			balanceHeap_deltas(perm, deltas, Q, query, maxT, totalLevels);
			}
		for(int i = maxT - 1; i; i--) {
			exchange(perm, 0, i);
			balanceHeap_deltas(perm, deltas, Q, query, i, __float2int_ru( log2f( (float) (2 * M + 1) ) ) );
			}
		
		// Initialize
		combinations[query] = 1;
		combinations_deltas[query] = deltas[perm[0] * Q + query];
		unsigned size = 1;
		
		unsigned minIndex;
		float minValue, value;
		uint64_t probe;
		unsigned maxPermIndex;
		
		// T includes original probing code, therefore start iterating from t=1
		for(int t = 1; t < T; t++) {
			minValue = FLT_MAX;
			for(int i = 0; i < size; i++) if((value = combinations_deltas[i * Q + query]) < minValue) {
				minIndex = i;
				minValue = value;
				}
			probe = combinations[minIndex * Q + query];
			int maxt = 64 - __clzll(probe);
			for(int i = 0; i < M; i++) queries[i * Q * T + query * T + t] = queries[i * Q * T + query * T];
			for(int i = 0; i < maxt; i++)
				if(probe & (1 << i)) {
					queries[ (perm[i] % M) * Q * T + query * T + t] += ((perm[i] >= M) ? 1 : (-1) );
					maxPermIndex = i;
					}
			if(maxPermIndex < maxT - 1) {
				probe |= ( 1 << (maxPermIndex + 1) );
				minValue += deltas[perm[maxPermIndex + 1] * Q + query];
				combinations[minIndex * Q + query] = probe;
				combinations_deltas[minIndex * Q + query] = minValue;
				
				probe -= (1 << maxPermIndex);
				minValue -= deltas[perm[maxPermIndex] * Q + query];
				combinations[size * Q + query] = probe;
				combinations_deltas[size * Q + query] = minValue;
				
				size++;
				}
			
			}
		}
}

/*
	Calculate delta values of queries based on their non-floored codes
	Also performs flooring in the end, to avoid an additional call to floorMatrix() when this function completes
	There are 2*M delta values for each query
		First M values refer to subtracting 1 from a dimension
		Last M values refer to adding 1 to a dimension
*/
__global__ void kernel_calculateDeltas(
	float* queries,	// non-floored queries' codes, [Q x M]
	const int Q,
	const int M,
	float* deltas	// [Q x 2*M]
	)
{
//	queries: [Q x M]
//	deltas: [Q x 2M], first M -> DOWN, last M -> UP

	int query = blockIdx.x * blockDim.x + threadIdx.x;
	int pos = blockIdx.y * blockDim.y + threadIdx.y;
//	if(query >= Q || pos >= M) return;
	if(query < Q && pos < M) {
		float value_nonFloored = queries[pos * Q + query];
		float value_floored = floorf(value_nonFloored);
		float deltaDown = value_nonFloored - value_floored;
		float deltaUp = 1 - deltaDown;
		
		queries[pos * Q + query] = value_floored;
		deltas[pos * Q + query] = deltaDown;
		deltas[(pos + M) * Q + query] = deltaUp;
		}
}
//####################################################################################################
/*
	Calculate distances
	Each row of a block refers to one distance calculation
		Each thread of row computes D / Bx sums of squares, where Bx is the number of threads in a row
	In the end, reduction is made and the total sum of squares is saved to <distances>
	The dimensions of this kernel's blocks is defined in cuLSH.h
		BLOCK_SIZE_REDUCE threads at the x dimension
		BLOCK_SIZE_Y threads at the y dimension
		[8 x 32] had the best performance results
*/
__global__ void kernel_calculateDistances(
	float* distances,	// [1 x totalIndices]
	const unsigned totalIndices,
	const float* queries,	// [D x Q]
	const int D,
	const int Q,
	const float* data,	// [D x N]
	const int N,
	const unsigned* candidateIndices,	// [[ 1 x totalIndices]
	const unsigned* queryIndices	// [1 x totalIndices]
)
{
	__shared__ float shared_distances[BLOCK_SIZE_Y][BLOCK_SIZE_REDUCE]; // blockDim.y * 32
	
	int index = blockIdx.y * blockDim.y + threadIdx.y;	// index to candidates
	float manhattan;
	float distance = 0.0;
	
	if(index < totalIndices) {
		int dataIndex = candidateIndices[index];
		int queryIndex = queryIndices[index];
		
		for(int column = threadIdx.x; column < D; column += BLOCK_SIZE_REDUCE) {
			manhattan = queries[queryIndex * D + column] - data[dataIndex * D + column];
			distance += manhattan * manhattan;
			}
		
		shared_distances[threadIdx.y][threadIdx.x] = distance;
		}

	__syncthreads();
	
		// Reduce
//		if(threadIdx.x < 64) shared_distances[threadIdx.y][threadIdx.x] += shared_distances[threadIdx.y][threadIdx.x + 64];
//		__syncthreads();
//		if(threadIdx.x < 32) shared_distances[threadIdx.y][threadIdx.x] += shared_distances[threadIdx.y][threadIdx.x + 32];
//		__syncthreads();
//		if(threadIdx.x < 16) shared_distances[threadIdx.y][threadIdx.x] += shared_distances[threadIdx.y][threadIdx.x + 16];
//		__syncthreads();
//		if(threadIdx.x < 8) shared_distances[threadIdx.y][threadIdx.x] += shared_distances[threadIdx.y][threadIdx.x + 8];
//		__syncthreads();
		if(threadIdx.x < 4) shared_distances[threadIdx.y][threadIdx.x] += shared_distances[threadIdx.y][threadIdx.x + 4];
		__syncthreads();
		if(threadIdx.x < 2) shared_distances[threadIdx.y][threadIdx.x] += shared_distances[threadIdx.y][threadIdx.x + 2];
		__syncthreads();
		if(!threadIdx.x) distances[index] = shared_distances[threadIdx.y][0] + shared_distances[threadIdx.y][1];
		
}
//####################################################################################################
/*
	Balance heap of distances after a new distance has been inserted
	When two distances are swapped in two <heap> positions, ids at the equivalent positions in <knnIds> are also swapped
	<heap> and <knnIds> have size [K x Q]
		In the end they contain distances and ids to the nearest neighbors of each query
*/
__device__ void balanceHeap_knnIds(
	float* heap,	//[K x Q]
	int* knnIds,	// [K x Q]
	const int& size,	// K * Q
	const unsigned& totalLevels	// total levels of heap tree
	)
{
	int parent = 0, left, right, maxIndex;
	for(int level = 0; level < totalLevels - 1; level++) {
			left = parent * 2 + 1;
			right = left + 1;
			/////
			
			float diff;
			maxIndex = parent;
			/*
				The additional comparison of ids in case distances are the same
				is made for testing purposes only. It ensures that in case
				two or more ids have the same distance, they will be assigned
				in ascending order of their id numbers.
				If this is not desired, comment that block and uncomment the
				next one, and performance will be slightly improved!
			*/
			if(left < size && (diff = heap[left] - heap[parent]) >= 0) {
				if(diff > 0 || (knnIds[left] > knnIds[parent])) maxIndex = left;
				}
			if(right < size && (diff = heap[right] - heap[maxIndex]) >= 0) {
				if(diff > 0 || (knnIds[right] > knnIds[maxIndex])) maxIndex = right;
				}
			/*
			maxIndex = parent;
			if(left < size && heap[left] > heap[parent])
				maxIndex = left;
			if(right < size && heap[right] >= heap[maxIndex])
				maxIndex = right;
			*/
			if(maxIndex == parent) break;
			exchange(heap, parent, maxIndex);
			exchange(knnIds, parent, maxIndex);
			parent = maxIndex;
			
		}
}

/*
	Calculate nearest neighbor ids after distances calculation has been completed
	The heap of distances is initialized to FLT_MAX
	Whenever a distance smaller than the heap's root is encountered,
		it replaces it and heap balancing is performed
*/
__global__ void kernel_configureIds(
	float* heap,	// [K x Q], will contain distances of nearest neighbors
	int* knnIds,	// [K x Q], will contain ids of nearest neighbors
	const int Q, const int K,
	const unsigned totalIndices,	// total number of candidates for all queries
	const float* distances,	// [1 x totalIndices]
	const unsigned* queryCandidates_totalIndices,	// total candidate ids
	const unsigned* queryCandidates_startingIndices,
	const unsigned* queryCandidates_sizes
	)
{
	int queryIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if(queryIndex < Q) {
		unsigned totalLevels = __float2uint_ru( log2f( (float) (K + 1) ) );
		unsigned startingIndex = queryCandidates_startingIndices[queryIndex];
		unsigned candidateSize = queryCandidates_sizes[queryIndex];
		float queryDistance;
		
		for(int i = candidateSize - 1; i >= 0; i--) {
		//for(int i = 0; i < candidateSize; i++) {
			if( (queryDistance = distances[startingIndex + i]) < heap[queryIndex * K] ) {
				heap[queryIndex * K] = queryDistance;
				knnIds[queryIndex * K] = queryCandidates_totalIndices[startingIndex + i];
				balanceHeap_knnIds(heap + queryIndex * K, knnIds + queryIndex * K, K, totalLevels);
				}
			}
		}
}

/*
	Sort nearest neighbor ids for each query according to their distances
	Heap calculations (<heap>, <knnIds>) have already been made
*/
__global__ void kernel_sortIds(
	float* heap,	// [K x Q]
	int* knnIds,	// [K x Q]
	const int Q,
	const int K
	)
{
	int queryIndex = blockIdx.x * blockDim.x + threadIdx.x;
	//if(queryIndex>=Q) return;
	if(queryIndex < Q) {
		for(int i = K - 1; i >= 1; i--) {
			exchange(heap + queryIndex * K, 0, i);
			exchange(knnIds + queryIndex * K, 0, i);
			unsigned totalLevels = __float2uint_ru( log2f( i + 1) );
			balanceHeap_knnIds(heap + queryIndex * K, knnIds + queryIndex * K, i, totalLevels);
			}
		}
}
//###########################################################################################################

}	// end of namespace

#endif	// #ifndef __cuLSH__Kernels__

