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
cuLSH::HashTables
	This is the structure used to perform indexing to a dataset.
	The user must first reset the structure by calling <reset>.
	Afterwards, indexing is performed by calling <index>.
<reset>:
	* bool reset(int N, int D, int L, int M, float W, FILE* debugStream = 0);
	* bool reset(int N, int D, int L, int M, float W, float** A, float** b, FILE* debugStream = 0);
	* bool reset(const char* filename, FILE* debugStream = 0);
	N: # of vectors in the dataset
	D: # of dimensions
	L: # of hash tables
	M: # of dimensions at projection space
	W: bucket width
	A: A[i] is the [D x M] projection matrix of table <i>
	b: b[i] is the [1 x M] projection vector of table <i>
	filename: file path to load a previously stored indexing structure
	debugStream: stream used to output debugging info
	TRUE is returned if structure was reseted successfully, FALSE otherwise
<index>:
	* bool index(const float* matrix, FILE* debugStream = 0);
	matrix: [D x N] dataset
	debugStream: stream used to output debugging info
	TRUE is returned if indexing was performed successfully, FALSE otherwise

*/

#ifndef __cuLSH__HashTables__
#define __cuLSH__HashTables__

namespace cuLSH {

class HashTables {
	private:
		int N, D;	// # of data vectors, # of dimensions at original space
		int L, M;	// # of tables, # of dimensions at projection space
		float W, **A, **b;	// bucket width, projection matrices [A] and vectors [b]
		// Buckets
		unsigned *buckets;	// L numbers of buckets, one per table
		unsigned **buckets_totalIndices;	// L vectors of [1 x N] total indices, one vector per table
		unsigned **buckets_startingIndices;	// L vectors of [1 x B] bucket starting indices, one vector per table (B is the # of buckets for one table)
		unsigned **buckets_sizes;	// L vectors of B sizes of buckets
		float **buckets_codes;	// L matrices of size [B x M] containing the bucket codes of each table
		// Memory management
		bool allocateMemory_Projection(void);	// allocate memory for [A], [b]
		bool allocateMemory_Indexing(void);	// allocate memory for buckets, and first level pointers of buckets_[totalIndices/startingIndices/sizes/codes]
		bool allocateMemory_Indexing(int table);	// allocate memory for one tables second level pointers of buckets_[totalIndices/startingIndices/sizes/codes]
		void freeMemory_Indexing(void);	// free memory of [A], [b]
		void freeMemory_Projection(void);	// free memory of buckets, buckets_[totalIndices/startingIndices/sizes/codes]
		
		// Check if parameters are all positive
		bool checkParameters() { return (N > 0 && D > 0 && L > 0 && M > 0 && W > 0.0); }
		
		// Generate random projection matrices [A] and vectors [b]
		bool generateRandomProjection(void);
		
	public:
		// Constructor / Destructor
		HashTables(void);
		~HashTables(void) { freeMemory_Indexing(); freeMemory_Projection(); }
		
		// Reset indexing structure
		bool reset(int N, int D, int L, int M, float W, FILE* debugStream = 0);
		bool reset(int N, int D, int L, int M, float W, float** A, float** b, FILE* debugStream = 0);
		bool reset(const char* filename, FILE* debugStream = 0);
		
		// Index matrix
		bool index(const float* matrix, FILE* debugStream = 0);
		
		// Get parameters
		int getN(void) { return N; }
		int getD(void) { return D; }
		int getL(void) { return L; }
		int getM(void) { return M; }
		float getW(void) { return W; }
		const float* getA(int table) { return (A ? A[table] : 0); }
		const float* getb(int table) { return (b ? b[table] : 0); }
		
		// Get buckets
		unsigned getBuckets(int table) { return (buckets ? buckets[table] : 0); }
		const unsigned* getBuckets(void) { return buckets; }
		const unsigned* getBuckets_totalIndices(int table) { return (buckets_totalIndices ? buckets_totalIndices[table] : 0); }
		const unsigned* getBuckets_startingIndices(int table) { return (buckets_startingIndices ? buckets_startingIndices[table] : 0); }
		const unsigned* getBuckets_sizes(int table) { return (buckets_sizes ? buckets_sizes[table] : 0); }
		const float* getBuckets_codes(int table) { return (buckets_codes ? buckets_codes[table] : 0); }
		
		// Save / Load indexing structure
		bool save(const char* filename, FILE* debugStream = 0);
		bool load(const char* filename, FILE* debugStream = 0);
	};

// CONSTRUCTOR
HashTables::HashTables(void)
{
	N = D = L = M = 0;
	W = 0.0;
	A = b = 0;
	buckets = 0;
	buckets_totalIndices = buckets_startingIndices = buckets_sizes = 0;
	buckets_codes = 0;
}

// GENERATE RANDOM PROJECTION MATRICES AND VECTORS
bool HashTables::generateRandomProjection(void)
{
	// Make sure parameters are all positive
	if(!checkParameters()) {
		DEF_PRINT_ERROR("Projection parameters must be positive! Have you reset the index structure properly???\n");
		return false;
		}
	// Make sure memory has been allocated for matrices [A] and vectors [b]
	if(!A || !b) {
		DEF_PRINT_ERROR("Pointers to matrices [A] and vectors [b] are null. Have you reset the index structure properly???\n");
		return false;
		}
	
	// Create number generators, matrix [A] and vector [b] on GPU
	curandGenerator_t generatorUniform;
	curandGenerator_t generatorNormal;
	ThrustFloatD d_A(D * M);
	ThrustFloatD d_b(M);
	
	// Initialize generators to uniform & normal
	if(curandCreateGenerator(&generatorUniform, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
		DEF_PRINT_ERROR("Failed to create uniform generator for [b] vectors...\n");
		return false;
		}
	if(curandCreateGenerator(&generatorNormal, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
		DEF_PRINT_ERROR("Failed to create normal generator for [A] matrices...\n");
		return false;
		}
	// Set random seed to generators
	if(curandSetPseudoRandomGeneratorSeed(generatorNormal, (unsigned long long) time(0)) != CURAND_STATUS_SUCCESS) {
		DEF_PRINT_ERROR("Failed to seed the generator for [A] matrices...\n");
		return false;
		}
	if(curandSetPseudoRandomGeneratorSeed(generatorUniform, (unsigned long long) time(0)) != CURAND_STATUS_SUCCESS) {
		DEF_PRINT_ERROR("Failed to seed the generator for [b] vectors...\n");
		return false;
		}
	// Produce matrix [A] and vector [b] for each table
	for(int table = 0; table < L; table++) {
		if(curandGenerateUniform(generatorUniform, thrust::raw_pointer_cast(d_b.data()), M) != CURAND_STATUS_SUCCESS) {
			DEF_PRINT_ERROR("Failed to generate uniformally distributed numbers for vector [b] of table %d/%d\n", table + 1, L);
			return false;
			}
		if(curandGenerateNormal(generatorNormal, thrust::raw_pointer_cast(d_A.data()), D * M, 0.0, 1.0) != CURAND_STATUS_SUCCESS) {
			DEF_PRINT_ERROR("Failed to generate normally distributed numbers for matrix [A] of table %d/%d\n", table + 1, L);
			return false;
			}
		float minb = *( thrust::min_element(d_b.begin(), d_b.end()) );
		thrust::transform(d_b.begin(), d_b.end(), thrust::make_constant_iterator(minb), d_b.begin(), thrust::minus<float>());
		thrust::transform(d_b.begin(), d_b.end(), thrust::make_constant_iterator(W), d_b.begin(), thrust::multiplies<float>());
		
		thrust::copy(d_A.begin(), d_A.end(), A[table]);
		thrust::copy(d_b.begin(), d_b.end(), b[table]);
		}
	// Destroy generators
	if(curandDestroyGenerator(generatorNormal) != CURAND_STATUS_SUCCESS) {
		DEF_PRINT_ERROR("Failed to destroy generator for [A] matrices...\n");
		return false;
		}
	if(curandDestroyGenerator(generatorUniform) != CURAND_STATUS_SUCCESS) {
		DEF_PRINT_ERROR("Failed to destroy generator for [b] vectors...\n");
		return false;
		}
	
	return true;
}

// RESET INDEX STRUCTURE
bool HashTables::reset(int N, int D, int L, int M, float W, FILE* debugStream)
{
	const char *funcString = "[HashTables::reset]";
	// Assign input parameters
	this->N = N;
	this->D = D;
	this->L = L;
	this->M = M;
	this->W = W;
	// Make sure parameters are all positive
	if(!checkParameters()) {
		DEF_PRINT_ERROR("Projection parameters must be positive! Have you reset the index structure properly???\n");
		return false;
		}
	// Allocate memory for projection and indexing objects, and generate random projection matrices [A] and vectors [b]
	if(!allocateMemory_Projection() || !generateRandomProjection()) return false;
	
	if(debugStream) fprintf(debugStream, "%s\tIndexing structure reset with parameters (N, D, L, M, W) = (%d, %d, %d, %d, %.2f), and random matrices [A] & vectors [b].\n", funcString, N, D, L, M, W);
	
	return true;
}

// RESET INDEX STRUCTURE
bool HashTables::reset(int N, int D, int L, int M, float W, float** A, float** b, FILE* debugStream)
{
	const char *funcString = "[HashTables::reset]";
	// Assign input parameters
	this->N = N;
	this->D = D;
	this->L = L;
	this->M = M;
	this->W = W;
	// Make sure parameters are all positive
	if(!checkParameters()) {
		DEF_PRINT_ERROR("Projection parameters must be positive! Have you reset the index structure properly???\n");
		return false;
		}
	// Allocate memory for projection and indexing objects, and assign input matrices [A] and vectors [b]
	if(!allocateMemory_Projection()) return false;
	for(int table = 0; table < L; table++) {
		/*
		memcpy((this->A)[table], A, D * M * sizeof(float));
		memcpy((this->b)[table], b, M * sizeof(float));
		*/
		thrust::copy_n(A[table], D * M, (this->A)[table]);
		thrust::copy_n(b[table], M, (this->b)[table]);
		}
	
	if(debugStream) fprintf(debugStream, "%s\tIndexing structure reset with parameters (N, D, L, M, W) = (%d, %d, %d, %d, %.2f), matrices [A] & vectors [b] have been manually set.\n", funcString, N, D, L, M, W);
	
	return true;
}

bool HashTables::reset(const char* filename, FILE* debugStream)
{
	const char *funcString = "[HashTables::reset]";
	if(debugStream) fprintf(debugStream, "%s\tLoading indexing structure from file \"%s\"...\n", funcString, filename);
	
	return load(filename, debugStream);
}

// ALLOCATE MEMORY FOR PROJECTION OBJECTS
bool HashTables::allocateMemory_Projection(void)
{
	// Make sure parameters are all positive
	if(!checkParameters()) {
		DEF_PRINT_ERROR("Projection parameters must be positive! Have you reset the index structure properly???\n");
		return false;
		}
	// In case memory is already allocated (for a previous projection, perhaps with different parameters), free it
	freeMemory_Projection();
	// Allocate memory for double pointers to projection objects
	if(!(A = (float**) malloc(L * sizeof(float*)))) {
		DEF_PRINT_ERROR("Memory allocation error for matrices A of all tables...\n");
		return false;
		}
	if(!(b = (float**) malloc(L * sizeof(float*)))) {
		DEF_PRINT_ERROR("Memory allocation error for vectors b of all tables...\n");
		return false;
		}
	// Allocate memory for matrices [A] and vectors[b] for each table
	for(int table = 0; table < L; table++) {
		if(!(A[table] = (float*) malloc(D * M * sizeof(float)))) {
			DEF_PRINT_ERROR("Memory allocation error for matrix A at table %d/%d\n", table + 1, L);
			return false;
			}
		if(!(b[table] = (float*) malloc(M * sizeof(float)))) {
			DEF_PRINT_ERROR("Memory allocation error for vector b at table %d/%d\n", table + 1, L);
			return false;
			}
		}
	
	return true;
}

// ALLOCATE MEMORY FOR INDEXING OBJECTS
bool HashTables::allocateMemory_Indexing(void)
{
	// Check if parameters are all positive
	if(!checkParameters()) {
		DEF_PRINT_ERROR("Projection parameters must be positive! Have you reset the index structure properly???\n");
		return false;
		}
	// In case memory is already allocated (for a previous indexing), free it
	freeMemory_Indexing();
	// Allocate memory for double pointers to indexing objects (and single pointers to # of buckets)
	if(!(buckets = (unsigned*) malloc(L * sizeof(unsigned)))) {
		DEF_PRINT_ERROR("Memory allocation failure for number of buckets of each table...\n");
		return false;
		}
	if(!(buckets_totalIndices = (unsigned**) malloc(L * sizeof(unsigned*)))) {
		DEF_PRINT_ERROR("Memory allocation error for total indices of buckets of each table...\n");
		return false;
		}
	if(!(buckets_startingIndices = (unsigned**) malloc(L * sizeof(unsigned*)))) {
		DEF_PRINT_ERROR("Memory allocation error for starting indices of buckets of each table...\n");
		return false;
		}
	if(!(buckets_sizes = (unsigned**) malloc(L * sizeof(unsigned*)))) {
		DEF_PRINT_ERROR("Memory allocation error for sizes of buckets of each table...\n");
		return false;
		}
	if(!(buckets_codes = (float**) malloc(L * sizeof(float*)))) {
		DEF_PRINT_ERROR("Memory allocation error for codes of buckets of each table...\n");
		return false;
		}
	// Assign zero to # of buckets, and null to single pointers to indexing objects (otherwise a segmentation fault might be producing when trying to free them (e.g. in destructor)
	thrust::fill(buckets, buckets + L, 0);
	thrust::fill(buckets_totalIndices, buckets_totalIndices + L, (unsigned*) 0);
	thrust::fill(buckets_startingIndices, buckets_startingIndices + L, (unsigned*) 0);
	thrust::fill(buckets_sizes, buckets_sizes + L, (unsigned*) 0);
	thrust::fill(buckets_codes, buckets_codes + L, (float*) 0);
	
	return true;
}

// ALLOCATE MEMORY FOR INDEXING OBJECTS FOR A SPECIFIC TABLE
bool HashTables::allocateMemory_Indexing(int table)
{
	// Make sure memory has been allocated for double pointer to indexing objects
	if(!buckets || !buckets_totalIndices || !buckets_startingIndices || !buckets_sizes || !buckets_codes) {
		DEF_PRINT_ERROR("Could not allocate memory for buckets for table %d/%d, has indexing memory allocation function been called for all tables first???\n", table + 1, L);
		return false;
		}
	// Get size of buckets for this table, and allocate memory for the other indexing objects
	int size = buckets[table];
	if(!(buckets_totalIndices[table] = (unsigned*) malloc(N * sizeof(unsigned)))) {
		DEF_PRINT_ERROR("Memory allocation error for total indices of buckets for table %d/%d\n", table + 1, L);
		return false;
		}
	if(!(buckets_startingIndices[table] = (unsigned*) malloc(size * sizeof(unsigned)))) {
		DEF_PRINT_ERROR("Memory allocation error for starting indices of buckets for table %d/%d\n", table + 1, L);
		return false;
		}
	if(!(buckets_sizes[table] = (unsigned*) malloc(size * sizeof(unsigned)))) {
		DEF_PRINT_ERROR("Memory allocation error for sizes of buckets for table %d/%d\n", table + 1, L);
		return false;
		}
	if(!(buckets_codes[table] = (float*) malloc(size * M * sizeof(float)))) {
		DEF_PRINT_ERROR("Memory allocation error for bucket codes for table %d/%d\n", table + 1, L);
		return false;
		}
	return true;
}

// FREE MEMORY FOR PROJECTION OBJECTS
void HashTables::freeMemory_Projection(void)
{
	if(A) {
		for(int table = 0; table < L; table++) free(A[table]);
		free(A);
		}
	if(b) {
		for(int table = 0; table < L; table++) free(b[table]);
		free(b);
		}
	A = b = 0;
}

// FREE MEMORY FOR INDEXING OBJECTS
void HashTables::freeMemory_Indexing(void)
{
	if(buckets) free(buckets);
	if(buckets_totalIndices) {
		for(int table = 0; table < L; table++) free(buckets_totalIndices[table]);
		free(buckets_totalIndices);
		}
	if(buckets_startingIndices) {
		for(int table = 0; table < L; table++) free(buckets_startingIndices[table]);
		free(buckets_startingIndices);
		}
	if(buckets_sizes) {
		for(int table = 0; table < L; table++) free(buckets_sizes[table]);
		free(buckets_sizes);
		}
	if(buckets_codes) {
		for(int table = 0; table < L; table++) free(buckets_codes[table]);
		free(buckets_codes);
		}
	buckets = 0;
	buckets_totalIndices = buckets_startingIndices = buckets_sizes = 0;
	buckets_codes = 0;
}

// SAVE TO FILE
bool HashTables::save(const char* filename, FILE* debugStream)
{
	const char *funcString = "[HashTables::save]";
	if(debugStream) fprintf(debugStream, "%s\tSaving indexing structure to file \"%s\"...\n", funcString, filename);
	
	// Check if projection and indexing objects are ok
	if(!A || !b) {
		DEF_PRINT_ERROR("Pointers to matrices [A] and/or vectors [b] are null. Have you reset the index structure properly???\n");
		return false;
		}
	if(!buckets || !buckets_totalIndices || !buckets_startingIndices || !buckets_sizes || !buckets_codes) {
		DEF_PRINT_ERROR("Pointers to bucket objects are null. Have you performed indexing???\n");
		return false;
		}
	// Create file
	FILE *fp;
	if(!(fp = fopen(filename, "wb"))) {
		DEF_PRINT_ERROR("Could not create file \"%s\" in order to store the hash tables...\n", filename);
		return false;
		}
	// Write parameters to file
	if(fwrite(&N, sizeof(int), 1, fp) != 1) return false;
	if(fwrite(&D, sizeof(int), 1, fp) != 1) return false;
	if(fwrite(&L, sizeof(int), 1, fp) != 1) return false;
	if(fwrite(&M, sizeof(int), 1, fp) != 1) return false;
	if(fwrite(&W, sizeof(float), 1, fp) != 1) return false;
	// Write projection objects to file
	for(int table = 0; table < L; table++) {
		if(fwrite(A[table], sizeof(float), D * M, fp) != D * M) return false;
		if(fwrite(b[table], sizeof(float), M, fp) != M) return false;
		}
	// Write bucket objects to file
	if(fwrite(buckets, sizeof(unsigned), L, fp) != L) return false;
	for(int table = 0; table < L; table++) {
		if(fwrite(buckets_totalIndices[table], sizeof(unsigned), N, fp) != N) return false;
		if(fwrite(buckets_startingIndices[table], sizeof(unsigned), buckets[table], fp) != buckets[table]) return false;
		if(fwrite(buckets_sizes[table], sizeof(unsigned), buckets[table], fp) != buckets[table]) return false;
		if(fwrite(buckets_codes[table], sizeof(float), buckets[table] * M, fp) != buckets[table] * M) return false;
		}
	
	if(ferror(fp)) return false;
	fclose(fp);
	return true;
}

// LOAD FROM FILE
bool HashTables::load(const char* filename, FILE* debugStream)
{
	const char *funcString = "[HashTables::load]";
	
	// Open file
	FILE *fp;
	if(!(fp = fopen(filename, "rb"))) {
		DEF_PRINT_ERROR("Could not open file \"%s\" in order to load the hash tables, or file does not exist...\n", filename);
		return false;
		}
	// Load parameters
	if(fread(&N, sizeof(int), 1, fp) != 1) return false;
	if(fread(&D, sizeof(int), 1, fp) != 1) return false;
	if(fread(&L, sizeof(int), 1, fp) != 1) return false;
	if(fread(&M, sizeof(int), 1, fp) != 1) return false;
	if(fread(&W, sizeof(float), 1, fp) != 1) return false;
	
	if(debugStream) fprintf(debugStream, "%s\tIndexing structure loaded from file \"%s\", (N, D, L, M, W) = (%d, %d, %d, %d, %.2f)\n", funcString, filename, N, D, L, M, W);
	
	// Allocate memory for projection objects
	if(!allocateMemory_Projection()) return false;
	// Load projection objects
	for(int table = 0; table < L; table++) {
		if(fread(A[table], sizeof(float), D * M, fp) != D * M) return false;
		if(fread(b[table], sizeof(float), M, fp) != M) return false;
		}
	// Allocate memory for indexing objects
	if(!allocateMemory_Indexing()) return false;
	// Load indexing objects
	if(fread(buckets, sizeof(unsigned), L, fp) != L) return false;
	for(int table = 0; table < L; table++) {
		// Allocate memory for indexing objects for this table
		if(!allocateMemory_Indexing(table)) return false;
		if(fread(buckets_totalIndices[table], sizeof(unsigned), N, fp) != N) return false;
		if(fread(buckets_startingIndices[table], sizeof(unsigned), buckets[table], fp) != buckets[table]) return false;
		if(fread(buckets_sizes[table], sizeof(unsigned), buckets[table], fp) != buckets[table]) return false;
		if(fread(buckets_codes[table], sizeof(float), buckets[table] * M, fp) != buckets[table] * M) return false;
		}
	
	if(ferror(fp)) return false;
	fclose(fp);
	return true;
}

// PERFORM INDEXING
bool HashTables::index(const float* matrix, FILE* debugStream)
{
	const char *funcString = "[HashTables::index]";
	// Make sure parameters are all positive
	if(!checkParameters()) {
		DEF_PRINT_ERROR("Parameters must be positive. Have you reset the index structure properly???\n");
		return false;
		}
	// Check if memory has been allocated for projection matrices [A] and vectors [b]
	if(!A || !b) {
		DEF_PRINT_ERROR("Pointers to projection matrices [A] and vectors [b] must be valid. Have you reset the index structure properly???\n");
		return false;
		}
	if(!allocateMemory_Indexing()) return false;
	
	// Perform indexing for each table
	for(int table = 0; table < L; table++) {
		if(debugStream) fprintf(debugStream, "%s\t*** Performing indexing at table %d/%d ***\n", funcString, table + 1, L);
		if(!indexTableData(
			&buckets[table],
			&buckets_totalIndices[table],
			&buckets_startingIndices[table],
			&buckets_sizes[table],
			&buckets_codes[table],
			matrix,
			A[table], b[table], W,
			N, D, M,
			debugStream
			)) {
				DEF_PRINT_ERROR("Indexing failed at table %d/%d\n", table + 1, L);
				return false;
				}
		}
	return true;
}

}	// end of namespace

#endif	// #ifndef __cuLSH__HashTables__
