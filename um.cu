#include "um.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <list>

#define N 1024

__global__ void kernel(int *a) {
	a[threadIdx.x]= threadIdx.x + 1;
}

int main() {
	UM umObj;
	int i;
	int *a = (int*)malloc(N*sizeof(int));

	/*Move this on Struct based on reflection (reflection ts)*/
	umObj.alocateMemObject<int, float>(N, N);

	int *d_a = (int *)umObj.getRefrence(0);
	float *d_a_1 = (float *)umObj.getRefrence(1);

	umObj.prerfetchOnDevice(0, N);

	umObj.prerfetchOnHost(1, N);

	umObj.setReadOnlyPrefer(1, N);

	umObj.setPreferAsHost(1, N);
	
	umObj.setPreferAsDevice(0, N);

	kernel<<<1, N>>>(d_a);

	cudaMemcpy(a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	for(i=0;i<N;i++) {
			printf("%d ",a[i]);
	}

	return 0;
}