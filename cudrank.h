#pragma once

#include<cstdio>

template<typename T>
T *cuda_ptr(T *host_ptr,size_t size)
{
	T* ptr;
	cudaMalloc(&ptr,size);
	cudaMemcpy(ptr,host_ptr,size,cudaMemcpyHostToDevice);
	return ptr;
}


template<typename T>
T *unified_ptr(size_t size)
{
	T* ptr;
	cudaMallocManaged(&ptr,size);
	return ptr;
}