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
#include "material.h"
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
color ray_color(ray &r,int depth,hittable_list **world,curandState* rand_state)
{	
	ray cur_ray =r;
	color cur_attenuation =vec3(1.0,1.0,1.0);
	for(int i =0;i<depth;i++)
	{
		hit_record rec;
		if((*world)->hit(cur_ray,0.001,infinity,rec))
		{
			ray scattered;
			vec3 attenuation;

			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered,rand_state))
			{
				cur_attenuation *=attenuation;
				cur_ray = scattered;
			}
			else
			{
				vec3(0.0,0.0,0.0);
			}
		}
		else
		{
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f*(unit_direction.y() + 1.0f);
			vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0,0.0,0.0);
}

__global__
void process(vec3 *final_out,int image_width,int image_height,int sample_size,int max_depth,hittable_list** world,camera *cam)
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
			pixel_color += ray_color(r,max_depth,world,&thread_rand_state);
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
        (*d_world)->add(new sphere(point3(0,0,-1), 0.5, new lambertian(vec3(0.8, 0.3, 0.3))));
		(*d_world)->add(new sphere(point3(0,-100.5,-1), 100, new lambertian(vec3(0.8, 0.8, 0.0))));
		(*d_world)->add(new sphere(point3(-1.0, 0.0, -1.0), 0.5,new metal(color(0.8, 0.8, 0.8),1.0)));
		(*d_world)->add(new sphere(point3( 1.0, 0.0, -1.0), 0.5,new metal(color(0.8, 0.8, 0.8),0)));
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
	const int max_depth = 25;




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
	process<<<num_blocks,block_size>>>(final_out,image_width,image_height,sample_size,max_depth,d_world,d_cam);
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