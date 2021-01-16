#include <iostream>
#include <math.h>
#include <stdio.h> 
#include "utility.h"
#include "cudrank.h"
#include "vec3.h"
#include "ray.h"
#include "color.h"
#include "hittable.h"
#include "sphere.h"
#include "hittable_list.h"


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
void process(vec3 *final_out,int image_width,int image_height,vec3 origin,vec3 horizontal,vec3 vertical,vec3 lower_left_corner,hittable_list** world)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for(int i=index;i<image_height*image_width;i+=stride)
	{
		int x = i%image_width;
		int y = (image_height-1) -(i/image_width);
		double u = double(x) / (image_width-1);
		double v = double(y) / (image_height-1);
		ray r(origin,lower_left_corner +u*horizontal+v*vertical - origin);
		color pixel_color = ray_color(r,world);
		write_color(final_out,i,pixel_color);
	}
}

__global__
void create_world(hittable **d_list, hittable_list **d_world)
 {
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		*d_world    = new hittable_list(d_list,10);
        (*d_world)->add(new sphere(vec3(0,0,-1), 0.5));
        (*d_world)->add(new sphere(vec3(0,-100.5,-1), 100));
        
    }
}


int main()
{

	// Image
	const double aspect_ratio = 16.0/9.0;
	const int image_height = 1080;
	const int image_width = static_cast<int>(image_height*aspect_ratio);
	vec3 *final_out = unified_ptr<vec3>(image_height*image_width*sizeof(vec3));

	// Camera
	double viewport_height = 2.0;
    double viewport_width = aspect_ratio * viewport_height;
    double focal_length = 1.0;

    vec3 origin = point3(0, 0, 0);
    vec3 horizontal = vec3(viewport_width, 0, 0);
    vec3 vertical = vec3(0, viewport_height, 0);
	vec3 lower_left_corner = origin - horizontal/2 - vertical/2 - vec3(0, 0, focal_length);


	//Kernel Parameters
	int block_size = 1024;
	int num_blocks = ceil(double(image_width*image_height))/double(block_size);



	//World
	hittable_list **d_world;
	cudaMalloc(&d_world, sizeof(hittable_list *));
	hittable **d_list;
	cudaMalloc(&d_list, 10*sizeof(hittable *));

    create_world<<<1,1>>>(d_list,d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	//Call Kernel
	process<<<num_blocks,block_size>>>(final_out,image_width,image_height,origin,horizontal,vertical,lower_left_corner,d_world);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	//File Handling
	FILE* file1 = fopen("image1.ppm","w");
	//Render

	fprintf(file1,"P3 %d %d\n255\n",image_width,image_height);

	for (int i = 0; i<image_width*image_height; i++)
	{
		fprintf(file1,"%d %d %d\n",(int) final_out[i][0],(int)final_out[i][1],(int)final_out[i][2]);
	}
	std::cerr<<"Done";
	cudaDeviceReset();
}