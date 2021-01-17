#include <iostream>
#include <math.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>
#include "utility.h"
#include "cudrank.h"
#include "vec3.h"
#include "ray.h"
#include "color.h"
#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}




__device__
color ray_color(ray &r,hittable_list **world)
{	
	hit_record rec;
    if ((*world)->hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

__global__
void process(vec3 *final_out,int image_width,int image_height,int sample_size,hittable_list** world,camera *cam)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	curandState thread_rand_state;
	curand_init(index,0,0,&thread_rand_state);
	for(int i=index;i<image_height*image_width;i+=stride)
	{

		int x = i%image_width;
		int y = (image_height-1) -(i/image_width);
		color pixel_color(0,0,0);
		for(int sample=0;sample<sample_size;sample++)
		{
			double u = double(x+random_double(&thread_rand_state)) / (image_width-1);
			double v = double(y+random_double(&thread_rand_state)) / (image_height-1);
	
			ray r = cam->get_ray(u,v);
			pixel_color += ray_color(r,world);
		}

		write_color(final_out,i,pixel_color,sample_size);
	}
}

__global__
void create_world(hittable **d_list, hittable_list **d_world)
 {
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{ 
		*d_world = new hittable_list(d_list,10);
        (*d_world)->add(new sphere(vec3(0,0,-1), 0.5));
        (*d_world)->add(new sphere(vec3(0,-100.5,-1), 100));
        
    }
}


__global__
void free_world(hittable_list **d_world)
{
	if(threadIdx.x == 0 && blockIdx.x==0)
	{
		delete (*d_world);
	}
}


int main()
{

	//Timer
	clock_t start,stop;
	start = clock();

	// Image
	const double aspect_ratio = 16.0/9.0;
	const int image_height = 1080;
	const int image_width = static_cast<int>(image_height*aspect_ratio);
	const int sample_size = 50;



	vec3 *final_out = unified_ptr<vec3>(image_height*image_width*sizeof(vec3));

	// Camera
	camera *h_cam = new camera();
	camera *d_cam = cuda_ptr<camera>(h_cam,sizeof(camera));
	delete h_cam;
	


	//Kernel Parameters
	int block_size = 256;
	int num_blocks = ceil(double(image_width*image_height))/double(block_size);



	//World
	hittable **d_list;
	cudaMalloc(&d_list, 10*sizeof(hittable *));
	hittable_list **d_world;
	cudaMalloc(&d_world, sizeof(hittable_list *));


    create_world<<<1,1>>>(d_list,d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	
	//Call Kernel
	process<<<num_blocks,block_size>>>(final_out,image_width,image_height,sample_size,d_world,d_cam);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	
	

    stop = clock();
	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	

	//File Handling and Importing ppm
	FILE* file1 = fopen("image1.ppm","w");

	fprintf(file1,"P3 %d %d\n255\n",image_width,image_height);

	for (int i = 0; i<image_width*image_height; i++)
	{
		fprintf(file1,"%d %d %d\n",(int) final_out[i][0],(int)final_out[i][1],(int)final_out[i][2]);
	}

	std::cerr<<"Done in "<<timer_seconds<<"s\n";
	//Free Memory

	free_world<<<1,1>>>(d_world);
	checkCudaErrors(cudaFree(d_list));
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_cam));
	checkCudaErrors(cudaFree(final_out));
	

	checkCudaErrors(cudaDeviceReset());
}