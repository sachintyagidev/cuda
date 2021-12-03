//File: um.h
#ifndef UM_H
#define UM_H

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <list>

using namespace std;

class UM {
	public:
      UM(bool isDefaultGPU = true) {
				IsDefaultGPU = isDefaultGPU;
      }

			~UM() {
				for (list<void *>::iterator it = allocatedMemRef.begin(); it != allocatedMemRef.end(); ++it){
					cudaFree(*it);
				}
      }

			template <typename T>
			__host__ void * alocateMem(int lenght) {
				void *x;
				cudaMallocManaged(&x, lenght*sizeof(T));
				
				/*All memory allocation will set prefer for GPU*/
				if (IsDefaultGPU) {
					int device = -1;
					cudaGetDevice(&device);
					cudaMemAdvise(x, lenght*sizeof(T), cudaMemAdviseSetPreferredLocation, device);
				}
				return x;
			}

			__host__ void alocateMemObject() {
			}

			template <typename T, typename... Rest>
			__host__ void alocateMemObject(T t, Rest... rest) {
				allocatedMemRef.push_back(alocateMem<T>(t));
				alocateMemObject(rest...);
			}

			__host__ void * getRefrence(int index) {
				list<void *>::iterator it = allocatedMemRef.begin();
				advance(it, index);
				return *it;
			}

			__host__ void prerfetchOnDevice(int index, int dataSize, cudaStream_t stream=NULL) {
				int device = -1;
  			cudaGetDevice(&device);
  			cudaMemPrefetchAsync(getRefrence(index), dataSize*sizeof(float), device, stream);
			}

			__host__ void prerfetchOnHost(int index, int dataSize, cudaStream_t stream=NULL) {
				cudaMemPrefetchAsync(getRefrence(index), dataSize*sizeof(float), cudaCpuDeviceId, stream);
			}

			__host__ void setReadOnlyPrefer(int index, int dataSize, cudaStream_t stream=NULL) {
				/*Read only data prefecth will improve performance*/
				cudaMemAdvise(getRefrence(index), dataSize, cudaMemAdviseSetReadMostly, 0);
				prerfetchOnDevice(index, dataSize, stream);
			}

			__host__ void setPreferAsHost(int index, int dataSize) {
				int device = -1;
  			cudaGetDevice(&device);
				/*As data preferred location is CPU then addding accessed by advice will set the mapping on the device*/
				cudaMemAdvise(getRefrence(index), dataSize, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
				cudaMemAdvise(getRefrence(index), dataSize, cudaMemAdviseSetAccessedBy, device);
			}

	private:
		list<void *> allocatedMemRef;
		bool IsDefaultGPU = true;
};

#endif