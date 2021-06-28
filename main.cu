#include <iostream>
#include <math.h>
#include <stdio.h>
#include <curand_kernel.h>
#include <time.h>
#include "utility.h"
#include "cudrank.h"
#include "vec3.h"
#include "onb.h"
#include "ray.h"
#include "color.h"
#include "textureMat.h"
#include "hittable.h"
#include "material.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
// #include "ppm_to_png.h"
#include "aarect.h"
#include "box.h"
#include "path.h"
#include "stationary_path.h"
#include "straight_path.h"
#include "circular_path.h"
#include "moving_sphere.h"
#include "raw_img.h"


// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"

#include "CImg.h"
using namespace cimg_library;

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
	// color background(0.70, 0.80, 1.00);
	color background(0,0,0);
	ray cur_ray =r;
	color cur_attenuation =vec3(1.0,1.0,1.0);
	for(int i =0;i<depth;i++)
	{
		hit_record rec;
		if((*world)->hit(cur_ray,0.001,infinity,rec))
		{
			ray scattered;
			vec3 albedo;
			color emitted = rec.mat_ptr->emitted(rec.u,rec.v,rec.p);
			double pdf = 1;
			
			if (rec.mat_ptr->scatter(cur_ray, rec, albedo, scattered,pdf,rand_state))
			{
				cur_attenuation = albedo*rec.mat_ptr->scattering_pdf(r, rec, scattered)*cur_attenuation/pdf + emitted;
				cur_ray = scattered;
			}
			else
			{
				return cur_attenuation*emitted;
			}
		}
		else
		{
			return cur_attenuation * background;
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
void create_world(hittable_list **d_world,moving_sphere **move_list,raw_img** image_list,int frames,double running_time)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{

		curandState thread_rand_state ;
		curand_init(2020,0,0,&thread_rand_state);
		*d_world = new hittable_list(500);
	

		

		auto red   = new lambertian(new solid_color(color(.65, .05, .05)));
		auto blue   = new lambertian(new solid_color(color(3.0/255.0, 129.0/255.0, 231.0/255.0)));
		auto white = new lambertian(new solid_color(color(.73, .73, .73)));
		auto green = new lambertian(new solid_color(color(.12, .45, .15)));
		auto sunlight = new diffuse_light(new solid_color(25*color(1,1,1)));
		auto moonlight = new lambertian(new solid_color(color(0.5, 0.5, 0.5)));
		auto white_metal = new metal(color(1,1,1),0.2);





		auto mercury_img = new lambertian(new image_texture(image_list[0]->data,image_list[0]->width,image_list[0]->height));
		auto venus_img = new lambertian(new image_texture(image_list[1]->data,image_list[1]->width,image_list[1]->height));
		auto earth_img = new lambertian(new image_texture(image_list[2]->data,image_list[2]->width,image_list[2]->height));
		auto mars_img = new lambertian(new image_texture(image_list[3]->data,image_list[3]->width,image_list[3]->height));
		auto jupiter_img = new lambertian(new image_texture(image_list[4]->data,image_list[4]->width,image_list[4]->height));
		auto saturn_img = new lambertian(new image_texture(image_list[5]->data,image_list[5]->width,image_list[5]->height));


		auto test_sphere = new sphere(point3(0,-50,0),2,sunlight);

		auto mercury_orbit = new circular_path(point3(0,0,0),point3(1,0,0),point3(0,0,1),20,0.5,frames,random_double(&thread_rand_state,0,2*pi));
		auto venus_orbit = new circular_path(point3(0,0,0),point3(1,0,0),point3(0,0,1),30,0.2,frames,random_double(&thread_rand_state,0,2*pi));
		auto earth_orbit = new circular_path(point3(0,0,0),point3(1,0,0),point3(0,0,1),50,0.05,frames,random_double(&thread_rand_state,0,2*pi));
		auto mars_orbit = new circular_path(point3(0,0,0),point3(1,0,0),point3(0,0,1),70,0.03,frames,random_double(&thread_rand_state,0,2*pi));
		auto jupiter_orbit = new circular_path(point3(0,0,0),point3(1,0,0),point3(0,0,1),100,0.01,frames,random_double(&thread_rand_state,0,2*pi));
		auto saturn_orbit = new circular_path(point3(0,0,0),point3(1,0,0),point3(0,0,1),120,0.01,frames,1.7*pi);
		auto earth_moon_orbit =  new circular_path(earth_orbit,point3(1,2,0),point3(0,0,1),8,1,frames,random_double(&thread_rand_state,0,2*pi));
		

		auto sun = new sphere(point3(0,0,0),12,sunlight);
		auto mercury = new moving_sphere(mercury_orbit, 2, mercury_img);
		auto venus = new moving_sphere(venus_orbit, 3, venus_img);
		auto earth = new moving_sphere(earth_orbit, 4, earth_img);
		auto mars = new moving_sphere(mars_orbit, 3, mars_img);
		auto jupiter = new moving_sphere(jupiter_orbit, 9, jupiter_img);
		auto saturn = new moving_sphere(saturn_orbit, 7, saturn_img);
		auto earth_moon = new moving_sphere(earth_moon_orbit, 1, moonlight);


		// auto mirror = new xz_rect(-400, 400, -400,400, -50, white_metal);
		// auto earth = new moving_sphere(earth_orbit, 4, blue);
 		// (*d_world)->add(mirror);

		(*d_world)->add(mercury);
		(*d_world)->add(venus);
		(*d_world)->add(earth);
		(*d_world)->add(mars);
		(*d_world)->add(jupiter);
		(*d_world)->add(saturn);
		(*d_world)->add(earth_moon);


		(*d_world)->add(test_sphere);

		(*d_world)->add(sun);
		move_list[0] = mercury;
		move_list[1] = venus;
		move_list[2] = earth;
		move_list[3] = mars;
		move_list[4] = jupiter;
		move_list[5] = saturn;
		move_list[6] = earth_moon;
    }
}

__global__
void move_world(moving_sphere **move_list)
{
	if(threadIdx.x == 0 && blockIdx.x==0)
	{
		move_list[0]->move();
		move_list[1]->move();
		move_list[2]->move();
		move_list[3]->move();
		move_list[4]->move();
		move_list[5]->move();
		move_list[6]->move();
	}
	
}

// __global__
// void create_world(hittable_list **d_world,moving_sphere **move_list,raw_img** image_list,int frames,double running_time)
//  {
// 	if (threadIdx.x == 0 && blockIdx.x == 0)
// 	{

// 		curandState thread_rand_state ;
// 		curand_init(2020,0,0,&thread_rand_state);
// 		*d_world = new hittable_list(500);
		

// 		auto red   = new lambertian(new solid_color(color(.65, .05, .05)));
// 		auto white = new lambertian(new solid_color(color(.73, .73, .73)));
// 		auto green = new lambertian(new solid_color(color(.12, .45, .15)));
// 		auto light = new diffuse_light(new solid_color(color(15, 15, 15)));
// 		auto white_metal = new metal(color(1,0.7,1),0);


// 		(*d_world)->add(new yz_rect(0, 555, 0, 555, 555, green));
// 		(*d_world)->add(new yz_rect(0, 555, 0, 555, 0, red));
// 		(*d_world)->add(new xz_rect(213, 343, 227, 332, 554, light));
// 		(*d_world)->add(new xz_rect(0, 555, 0, 555, 0, white));
// 		(*d_world)->add(new xz_rect(0, 555, 0, 555, 555, white));
// 		(*d_world)->add(new xy_rect(0, 555, 0, 555, 555, white));	
// 		(*d_world)->add(new box(point3(130, 0, 65), point3(295, 165, 230), white));
// 		(*d_world)->add(new box(point3(265, 0, 295), point3(430, 330, 460), white));
//     }
// }


// __global__
// void move_world(moving_sphere **move_list)
// {
// 	if(threadIdx.x == 0 && blockIdx.x==0)
// 	{

// 	}
	
// }

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


	// Image
	const double aspect_ratio = 16.0/9.0;
	const int image_height = 720;
	const int image_width = static_cast<int>(image_height*aspect_ratio);
	const int sample_size = 2000;
	const int max_depth = 50;
	const int fps = 25;
	const double running_time = 10;
	const int frames = fps*running_time;





	vec3 *final_out = unified_ptr<vec3>(image_height*image_width*sizeof(vec3));

	// Camera
	point3 lookfrom(200,200,100);
	// point3 lookfrom(200,0,0);

    point3 lookat(0, 0, 0);
    vec3 vup(0,1,0);
	// point3 lookfrom(278, 278, -800);
    // point3 lookat(278, 278, 0);
    // vec3 vup(0,1,0);

	
	// camera *h_cam = new camera(lookfrom, lookat, vup, 40, aspect_ratio);
	// camera *d_cam = cuda_ptr<camera>(h_cam,sizeof(camera));
	// delete h_cam;
	
	camera *h_cam = new camera(lookfrom, lookat, vup, 40, aspect_ratio);
	camera *d_cam = cuda_ptr<camera>(h_cam,sizeof(camera));
	delete h_cam;
	


	//Kernel Parameters
	int block_size = 512;
	int num_blocks = ceil(double(image_width*image_height))/double(block_size);



	//World
	
	hittable_list **d_world;
	cudaMalloc(&d_world, sizeof(hittable_list *));

	moving_sphere **move_list;
	cudaMalloc(&move_list,10*sizeof(moving_sphere *));

	raw_img** image_list;
	cudaMallocManaged(&image_list,10*sizeof(raw_img *));


	// Image Data
	



	CImg<unsigned char> mercury_img("data/mercurymap.jpg");
	image_list[0] = cuda_ptr<raw_img>(new raw_img(cuda_ptr<unsigned char>(mercury_img.data(),static_cast<size_t>(sizeof(unsigned char)*mercury_img.width()*mercury_img.height()*3)),mercury_img.width(),mercury_img.height()),sizeof(raw_img)) ;

	CImg<unsigned char> venus_img("data/venusmap.jpg");
	image_list[1] = cuda_ptr<raw_img>(new raw_img(cuda_ptr<unsigned char>(venus_img.data(),static_cast<size_t>(sizeof(unsigned char)*venus_img.width()*venus_img.height()*3)),venus_img.width(),venus_img.height()),sizeof(raw_img)) ;

	CImg<unsigned char> earth_img("data/earthmap.jpg");
	image_list[2] = cuda_ptr<raw_img>(new raw_img(cuda_ptr<unsigned char>(earth_img.data(),static_cast<size_t>(sizeof(unsigned char)*earth_img.width()*earth_img.height()*3)),earth_img.width(),earth_img.height()),sizeof(raw_img)) ;

	CImg<unsigned char> mars_img("data/marsmap.jpg");
	image_list[3] = cuda_ptr<raw_img>(new raw_img(cuda_ptr<unsigned char>(mars_img.data(),static_cast<size_t>(sizeof(unsigned char)*mars_img.width()*mars_img.height()*3)),mars_img.width(),mars_img.height()),sizeof(raw_img)) ;

	CImg<unsigned char> jupiter_img("data/jupitermap.jpg");
	image_list[4] = cuda_ptr<raw_img>(new raw_img(cuda_ptr<unsigned char>(jupiter_img.data(),static_cast<size_t>(sizeof(unsigned char)*jupiter_img.width()*jupiter_img.height()*3)),jupiter_img.width(),jupiter_img.height()),sizeof(raw_img)) ;

	CImg<unsigned char> saturn_img("data/saturnmap.jpg");
	image_list[5] = cuda_ptr<raw_img>(new raw_img(cuda_ptr<unsigned char>(saturn_img.data(),static_cast<size_t>(sizeof(unsigned char)*saturn_img.width()*saturn_img.height()*3)),saturn_img.width(),saturn_img.height()),sizeof(raw_img)) ;








	// if (!data) 
	// {
	// 	std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
	// 	width = height = 0;
	// }

    create_world<<<1,1>>>(d_world,move_list,image_list,fps,running_time);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	clock_t start,stop;
	start = clock();

	
	//Call Kernel
	
	char str[5];
	char* file_start = "outputppm/image";
	char* extension = ".ppm";
	char file_name[30];



	for(int j =0;j<frames;j++)
	{
		process<<<num_blocks,block_size>>>(final_out,image_width,image_height,sample_size,max_depth,d_world,d_cam);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		//File Handling and Importing ppm
		sprintf(str, "%d", j);
		strcpy(file_name,file_start);
		strcat(file_name,str);
		strcat(file_name,extension);

		FILE* file1 = fopen(file_name,"w");
	
		fprintf(file1,"P3 %d %d\n255\n",image_width,image_height);
	
		for (int i = 0; i<image_width*image_height; i++)
		{
			fprintf(file1,"%d %d %d\n",static_cast<int>(final_out[i][0]),static_cast<int>(final_out[i][1]),static_cast<int>(final_out[i][2]));
		}
	
		fclose(file1);
		move_world<<<1,1>>>(move_list);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		std::cerr<<(double(j+1)/double(frames))*100<<"%"<<"\n"<<std::flush;
	
	}




    stop = clock();
	double timer_seconds = ((static_cast<double>(stop - start))) / CLOCKS_PER_SEC;

	std::cerr<<"Done in "<<timer_seconds<<"s\n";
	//Free Memory

	free_world<<<1,1>>>(d_world);
	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_cam));
	checkCudaErrors(cudaFree(final_out));
	checkCudaErrors(cudaDeviceReset());
	checkCudaErrors(cudaDeviceSynchronize());


}