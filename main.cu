#include <iostream>
#include <math.h>
#include "cudrank.h"
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include <stdio.h> 



__device__ __host__
double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    double a = dot(r.direction(), r.direction());
    double b = 2.0 * dot(oc, r.direction());
    double c = dot(oc, oc) - radius*radius;
	double discriminant = b*b - 4*a*c;
	if (discriminant < 0)
	{
        return -1.0;
	}
	else
	{
        return (-b - sqrt(discriminant) ) / (2.0*a);
    }
}
__device__ __host__
color ray_color(const ray& r)
{	
	double t = hit_sphere(point3(0,0,-1),0.5,r);
	if(t>0.0)
	{
		vec3 N = unit_vector(r.at(t) - vec3(0,0,-1));
        return 0.5*color(N.x()+1, N.y()+1, N.z()+1);
	}
	vec3 unit_direction = unit_vector(r.direction());
	t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

__global__
void process(vec3 *final_out,int image_width,int image_height,vec3 origin,vec3 horizontal,vec3 vertical,vec3 lower_left_corner)
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
		color pixel_color = ray_color(r);
		write_color(final_out,i,pixel_color);
	}
}

int main()
{
	
	#ifndef ONLINE_JUDGE
		freopen("image1.ppm", "w", stdout);
	#endif

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


	//Call Kernel
	process<<<num_blocks,block_size>>>(final_out,image_width,image_height,origin,horizontal,vertical,lower_left_corner);
	cudaDeviceSynchronize();


	//File Handling
	FILE* file1 = fopen("image1.ppm","w");
	//Render
	fprintf(file1,"P3 %d %d\n255\n",image_width,image_height);
	// std::cout<<"P3\n"<<image_width<<" "<<image_height<<"\n255\n";
	for (int i = 0; i<image_width*image_height; i++)
	{
		fprintf(file1,"%d %d %d\n",(int)final_out[i][0],(int)final_out[i][1],(int)final_out[i][2]);
	}
	std::cerr<<"Done";
}