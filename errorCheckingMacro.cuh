#ifndef CHECK_CUDA_ERROR_M_H
#define CHECK_CUDA_ERROR_M_H
 
#define PRINT_ON_SUCCESS 1
 
// To be used around calls that return an error code, ex. cudaDeviceSynchronize or cudaMallocManaged
void checkError(cudaError_t code, char const * func, const char *file, const int line, bool abort = true);
#define checkCUDAError(val) { checkError((val), #val, __FILE__, __LINE__); }    // in-line regular function
 
// To be used after calls that do not return an error code, ex. kernels to check kernel launch errors
void checkLastError(char const * func, const char *file, const int line, bool abort = true);
#define checkLastCUDAError(func) { checkLastError(func, __FILE__, __LINE__); }
#define checkLastCUDAError_noAbort(func) { checkLastError(func, __FILE__, __LINE__, 0); }
 
#endif // CHECK_CUDA_ERROR_M_H