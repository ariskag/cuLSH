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

/*
cuLSH::SearchTables
	This is the structure used to perform searching after indexing has been made.
	The user must reset the structure by calling <reset>.
	Afterwards, searching is performed by calling <search>.
<reset>:
	* void reset(HashTables* hashtables, int K, int T = 1);
	hashtables: pointer to indexing structure containing projection parameters and buckets
	K: # of nearest neighbors to be returned at searching
	T: # of total probing bins to be examined for each query (T = 1 for classic LSH)
<search>:
	* bool search(const float* queries, const int Q, const float* dataset, FILE* debugStream = 0);
	queries: [D x Q] matrix of queries
	Q: # of query vectors
	dataset: [D x N] dataset, must be the same that was used with indexing structure hashtables
	debugStream: stream used to output debugging info
*/

#ifndef __cuLSH__SearchTables__
#define __cuLSH__SearchTables__

namespace cuLSH {

class SearchTables {
	public:
		typedef struct {
			float buckets;	// time elapsed to find queries' matching buckets
			float collect;	// time elapsed to collect the candidates from queries' matching buckets
			float sort;	// time elapsed to sort candidate indices
			float unique;	// time elapsed to extract unique candidate incices
			float distances;	// time elapsed to calculate distances
			float knn;	// time elapsed to extract K nearest neighbors
			void reset(void) { buckets = collect = sort = unique = distances = knn = 0.0; }
			} timesStruct;
	private:
		int Q;	// # of queries
		int L;	// # of tables
		int K, T;	// # of nearest neighbors, # of total probing bins
		HashTables *hashtables;	// pointer to indexing structure
		timesStruct times;	// time values recorded at searching
		
		int **queryBuckets; // L matrices of size [T x Q], indices to queries' matching buckets
		
		int *knn_ids;	// [K x Q] indices to K nearest neighbors
		float *knn_distances;	// [K x Q] distances of K nearest neighbors
		
		unsigned totalCandidates_unique;	// total unique candidates for all queries
		unsigned totalCandidates_nonUnique;	// total non-unique candidates for all queries
		
		// Memory management
		bool allocateMemory(void);	// allocate memory for queryBuckets, knn_ids, knn_distances
		void freeMemory(void); // free allocated memory
	public:
		// Constructor / Destructor
		SearchTables(void);
		~SearchTables(void) { freeMemory(); }
		
		// Reset structure
		void reset(HashTables* hashtables, int K, int T = 1);
		
		// Perform searching
		bool search(const float* queries, const int Q, const float* dataset, FILE* debugStream = 0);
		
		// Get selectivity
		float getSelectivity(void) { return ( (hashtables && hashtables->getN() && Q) ? ( (totalCandidates_unique / (float) (hashtables->getN()) / (float) Q) ) : 0 ); }
		
		// Get percentage of non-unique candidates
		float getNonUniquePercentage(void) { return (totalCandidates_unique ? ((totalCandidates_nonUnique - totalCandidates_unique) / (float) totalCandidates_unique) : 0); }
		
		// Get objects
		const int* getQueryBuckets(int table) { return (queryBuckets ? queryBuckets[table] : 0); }
		unsigned getTotalCandidates_unique(void) { return totalCandidates_unique; }
		unsigned getTotalCandidates_nonUnique(void) { return totalCandidates_nonUnique; }
		
		// Get ids and distances of nearest neighbors
		const int* getKnnIds(void) { return knn_ids; }
		const float* getKnnDistances(void) { return knn_distances; }
		
		// Get searching times
		void getTimes(timesStruct* times) {
			times->buckets = this->times.buckets;
			times->collect = this->times.collect;
			times->sort = this->times.sort;
			times->unique = this->times.unique;
			times->distances = this->times.distances;
			times->knn = this->times.knn;
			}
		
	};

//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

// CONSTRUCTOR
SearchTables::SearchTables(void)
{
	Q = L = K = T = 0;
	
	totalCandidates_unique = totalCandidates_nonUnique = 0;	// just in case getNonUniquePercentage() is called before searching
	
	hashtables = 0;
	queryBuckets = 0;
	knn_ids = 0;
	knn_distances = 0;
}

//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

// RESET SEARCHING STRUCTURE
void SearchTables::reset(HashTables* hashtables, int K, int T)
{
	this->hashtables = hashtables;
	L = hashtables->getL();
	this->K = K;
	this->T = T;
	
	freeMemory();
	
	totalCandidates_unique = totalCandidates_nonUnique = 0;	// just in case getNonUniquePercentage() is called before searching
	knn_ids = 0;
	knn_distances = 0;
}

//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

// ALLOCATE MEMORY
bool SearchTables::allocateMemory(void)
{
	if(!hashtables) {
		DEF_PRINT_ERROR("No indexing structure has been associated with searching structure. Have you reset the searching structure properly???\n");
		return false;
		}
	if(!Q || !L || !K || !T) {
		DEF_PRINT_ERROR("Integer parameters must be all positive. Have you reset the indexing structure properly before assigning it to this searching structure???\n");
		return false;
		}
	
	freeMemory();
	
	// queryBuckets
	if(!(queryBuckets = (int**) malloc(L * sizeof(int*)))) {
		DEF_PRINT_ERROR("Could not allocate memory for queries' matching buckets for all tables...\n");
		return false;
		}
	for(int table = 0; table < L; table++)
		if(!(queryBuckets[table] = (int*) malloc(Q * T * sizeof(int)))) {
			DEF_PRINT_ERROR("Could not allocate memory for queries' matching buckets for table %d/%d\n", table + 1, L);
			return false;
			}
	// knn_ids
	if(!(knn_ids = (int*) malloc(Q * K * sizeof(unsigned)))) {
		DEF_PRINT_ERROR("Could not allocate memory for queries' knn ids...\n");
		return false;
		}
	// knn_distances
	if(!(knn_distances = (float*) malloc(Q * K * sizeof(unsigned)))) {
		DEF_PRINT_ERROR("Could not allocate memory for queries' knn distances...\n");
		return false;
		}
	
	return true;
}

//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

// FREE MEMORY
void SearchTables::freeMemory(void)
{
	if(queryBuckets) {
		for(int table = 0; table < L; table++) free(queryBuckets[table]);
		free(queryBuckets);
		}
	free(knn_ids);
	free(knn_distances);
}

//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

// PERFORM SEARCHING
bool SearchTables::search(const float* queries, const int Q, const float* dataset, FILE* debugStream)
{
	const char *funcString = "[SearchTables::search]";
	
	cudaEvent_t event_start;
	cudaEvent_t event_stop;
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);
	float event_ms;
	clock_t clock_start;
	clock_t clock_stop;
	times.reset();
	
	// Make sure an indexing structure has been associated
	if(!hashtables) {
		DEF_PRINT_ERROR("No indexing structure has been associated with searching structure...\n");
		return false;
		}
	// Assign number of queries to structure
	this->Q = Q;
	// Allocate memory for queryBuckets, candidates_[startingIndices/sizes]
	if(!allocateMemory()) return false;
	
	totalCandidates_nonUnique = 0;
	totalCandidates_unique = 0;
	
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	// First, calculate queries' buckets, and determine silmutaneously total number of non-unique candidates for each query
	//*****************************************************
	ThrustUnsignedH h_candidates_sizes_nonUnique(Q, 0);
	ThrustUnsignedH h_candidates_sizes(0);
	ThrustUnsignedH h_candidates_starting(0);
	ThrustUnsignedH h_candidates_total(0);
	//*****************************************************
	
//cudaProfilerStart();
	cudaEventRecord(event_start);
	for(int table = 0; table < L; table++) {
		if(debugStream) fprintf(debugStream, "%s\t*** Searching for queries' matching buckets at table %d/%d ***\n", funcString, table + 1, L);
		// Find queries' matching buckets for this table
		if(!findTableQueryBins(
			queryBuckets[table], queries,
			hashtables->getA(table),
			hashtables->getb(table),
			hashtables->getW(), Q, hashtables->getD(), hashtables->getM(), T,
			hashtables->getBuckets_codes(table),
			hashtables->getBuckets(table),
			debugStream
			)) {
				DEF_PRINT_ERROR("Failed to find queries' matching bins...\n");
				return false;
				}
		// Add to non-unique candidate size for each query the size of its candidates for this table
		const unsigned *buckets_sizes = hashtables->getBuckets_sizes(table);
		for(int query = 0; query < Q; query++) {
			for(int probe = 0; probe < T; probe++)
				if(queryBuckets[table][query * T + probe]!=-1)
					h_candidates_sizes_nonUnique[query] += buckets_sizes[queryBuckets[table][query * T + probe]];
			}
		}
	cudaEventRecord(event_stop);
	cudaEventSynchronize(event_stop);
	cudaEventElapsedTime(&event_ms, event_start, event_stop);
	times.buckets = event_ms / 1000.0;
//cudaProfilerStop();
	
	// Determine number of all (non-unique) candidates for all queries
	totalCandidates_nonUnique = thrust::reduce( h_candidates_sizes_nonUnique.begin(), h_candidates_sizes_nonUnique.end() );
	//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
	int N = hashtables->getN();
	int D = hashtables->getD();
	//#########################
	ThrustFloatD d_dataset(dataset, dataset + D * N);
	ThrustFloatD d_queries(queries, queries + D * Q);
	//#########################
	ThrustIntD d_knnIds(0);
	ThrustUnsignedD d_startingIndices(0);
	ThrustUnsignedD d_sizes(0);
	//#########################
	ThrustUnsignedD d_totalIndices(0);
	ThrustFloatD d_distances(0);
	//#########################
	
	if(debugStream) fprintf(debugStream, "%s\tPerforming searching. (Q, K, T) = (%d, %d, %d)\n", funcString, Q, K, T);
	
	// Get free GPU memory
	size_t gpuMemory_free, gpuMemory_total;
	cuMemGetInfo(&gpuMemory_free, &gpuMemory_total);
	
	size_t sizePerCandidate = 2 * sizeof(unsigned) + sizeof(float);	// because of d_candidates_totalIndices, d_candidates2queries, d_distances
	size_t sizePerQuery = K * (sizeof(float) + sizeof(unsigned)) + 2 * sizeof(unsigned);
	
	if(debugStream)
		fprintf(debugStream, "%s\tFree GPU memory: %.2fMB...\n", funcString, DEF_MB(gpuMemory_free));
	
	int firstQuery = 0, lastQuery;
	
cudaProfilerStart();
	
	while(firstQuery < Q) {
		size_t totalSize = 0;
		for(lastQuery = firstQuery; lastQuery < Q; lastQuery++)
			if(totalSize + sizePerQuery + sizePerCandidate * h_candidates_sizes_nonUnique[lastQuery] > gpuMemory_free - 20 * (1 << 20)) break;
			//else if(lastQuery == firstQuery + 100) break;
			else totalSize += sizePerQuery + sizePerCandidate * h_candidates_sizes_nonUnique[lastQuery];
		
		if(firstQuery == lastQuery) {
			DEF_PRINT_ERROR("Cannot load all candidates for computation of knn ids of query %d/%d...\n", firstQuery+1, Q);
			return false;
			}
		
		unsigned blockCandidates = thrust::reduce(h_candidates_sizes_nonUnique.begin() + firstQuery, h_candidates_sizes_nonUnique.begin() + lastQuery);
		unsigned blockQueries = lastQuery - firstQuery;
		
		if(debugStream) fprintf(debugStream, "%s\tCalculating for queries [%d, %d) (%d queries, %.2fM non-unique candidates, %.2fMB)\n", funcString, firstQuery, lastQuery, blockQueries, blockCandidates / (float) 1000000, DEF_MB(totalSize));
		
		h_candidates_sizes.resize(blockQueries);
		h_candidates_starting.resize(blockQueries);
		h_candidates_total.resize(blockCandidates);
		
		thrust::fill(h_candidates_sizes.begin(), h_candidates_sizes.end(), 0);
		
		////////////////
		unsigned totalSoFar = 0;
		for(int query = 0; query < blockQueries; query++) {
			h_candidates_starting[query] = totalSoFar;
			clock_start = clock();
			for(int table = 0; table < L; table++) {
				const unsigned *buckets_totalIndices = hashtables->getBuckets_totalIndices(table);
				const unsigned *buckets_startingIndices = hashtables->getBuckets_startingIndices(table);
				const unsigned *buckets_sizes = hashtables->getBuckets_sizes(table);
				
				for(int probe = 0; probe < T; probe++) {
					int queryBucket = queryBuckets[table][(query + firstQuery) * T + probe];
					if(queryBucket != -1) {
						thrust::copy_n(
							buckets_totalIndices + buckets_startingIndices[queryBucket],
							buckets_sizes[queryBucket],
							h_candidates_total.begin() + h_candidates_starting[query] + h_candidates_sizes[query]
							);
						h_candidates_sizes[query] += buckets_sizes[queryBucket];
						}
					}	// end of probe loop
				}	// end of table loop
			clock_stop = clock();
			times.sort += DEF_SECONDS(clock_stop - clock_start);
			
			// Sort queries' candidates
			clock_start = clock();
			thrust::sort(
				h_candidates_total.begin() + h_candidates_starting[query],
				h_candidates_total.begin() + h_candidates_starting[query] + h_candidates_sizes[query]
				);
			clock_stop = clock();
			times.collect += DEF_SECONDS(clock_stop - clock_start);
			// Distinguish unique queries' candidates
			clock_start = clock();
			h_candidates_sizes[query] = thrust::distance(
				h_candidates_total.begin() + h_candidates_starting[query],
				thrust::unique(
					h_candidates_total.begin() + h_candidates_starting[query],
					h_candidates_total.begin() + h_candidates_starting[query] + h_candidates_sizes[query]
					)
				);
			clock_stop = clock();
			times.unique += DEF_SECONDS(clock_stop - clock_start);
			totalSoFar += h_candidates_sizes[query];
			}	// end of query loop
		////////////////
		blockCandidates = totalSoFar;
		totalCandidates_unique += blockCandidates;
		h_candidates_total.resize(blockCandidates);
		////////////////
		d_knnIds.resize(K * blockQueries);
		d_startingIndices.resize(blockQueries);
		d_sizes.resize(blockQueries);
		////////////////
		d_totalIndices.resize(blockCandidates);
		d_distances.resize(blockCandidates);
		////////////////
		thrust::copy_n(h_candidates_starting.begin(), blockQueries, d_startingIndices.begin());
		thrust::copy_n(h_candidates_sizes.begin(), blockQueries, d_sizes.begin());
		thrust::copy_n(h_candidates_total.begin(), blockCandidates, d_totalIndices.begin());
		////////////////
		
		ThrustUnsignedD d_candidates2queries(blockCandidates);
		
		for(int query = 0; query < blockQueries; query++)
			thrust::fill_n(
				d_candidates2queries.begin() + d_startingIndices[query],
				h_candidates_sizes[query],//d_sizes[query],
				query + firstQuery
				);
		
//		for(int query = 0; query < blockQueries; query++) printf("[%d](start=%d, size=%d)\t", query, h_candidates_starting[query], h_candidates_sizes[query]);
//		printf("\n");
//		ThrustUnsignedH h_c2q(d_candidates2queries);
//		for(int i = 0; i < h_c2q.size()-1; i++) if(h_c2q[i]!=h_c2q[i+1]) printf("(%u->%u)@%d\t", h_c2q[i], h_c2q[i+1], i);
		
		cudaEventRecord(event_start);
		calculateDistances(
			d_distances,
			d_queries,
			D, Q,
			d_dataset,
			N,
			d_totalIndices,
			d_candidates2queries,
			debugStream
			);
		cudaEventRecord(event_stop);
		cudaEventSynchronize(event_stop);
		cudaEventElapsedTime(&event_ms, event_start, event_stop);
		times.distances += event_ms / 1000.0;
		
		/*
		calculateDistances2(
			d_distances,
			d_queries,
			D, Q,
			d_dataset,
			N,
			h_candidates_starting[firstQuery],
			firstQuery,
			lastQuery,
			d_totalIndices,
			d_startingIndices,
			debugStream
			);
		*/
		
		cudaEventRecord(event_start);
		calculateIds(
			d_knnIds,
			d_distances,
			blockQueries, K,
			d_totalIndices,
			d_startingIndices,
			d_sizes
			);
		cudaEventRecord(event_stop);
		cudaEventSynchronize(event_stop);
		cudaEventElapsedTime(&event_ms, event_start, event_stop);
		times.knn += event_ms / 1000.0;
		
		thrust::copy_n(d_knnIds.begin(), K * blockQueries, knn_ids + K * firstQuery);
		thrust::copy_n(d_distances.begin(), K * blockQueries, knn_distances + K * firstQuery);
		
		d_distances.clear(); d_distances.shrink_to_fit();
		d_totalIndices.clear(); d_totalIndices.shrink_to_fit();
		d_sizes.clear(); d_sizes.shrink_to_fit();
		d_startingIndices.clear(); d_startingIndices.shrink_to_fit();
		d_knnIds.clear(); d_knnIds.shrink_to_fit();
		
		firstQuery = lastQuery;
		}
	
cudaProfilerStop();
	
	printf("Non-unique: %u, unique: %u, selectivity: %.2f%%\n", totalCandidates_nonUnique, totalCandidates_unique, getSelectivity() * 100.0);
//	if(debugStream) fprintf(debugStream, "%s\t-----> %f seconds to calculate knn ids...\n", funcString, DEF_SECONDS(clock() - t));
	
	cudaEventDestroy(event_start);
	cudaEventDestroy(event_stop);
	
	return true;
}

}	// end of namespace

#endif	// #ifndef __cuLSH__SearchTables__

