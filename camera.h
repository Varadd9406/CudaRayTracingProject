#pragma once


class camera
{
	public:
	__device__ __host__
	camera(point3 lookfrom,point3 lookat,vec3 vup,double vfov, double aspect_ratio)
	{
    	double theta = degrees_to_radians(vfov);
    	double h = tan(theta/2);
    	double viewport_height = 2.0 * h;
    	double viewport_width = aspect_ratio * viewport_height;

		vec3 w = unit_vector(lookfrom - lookat);
		vec3 u = unit_vector(cross(vup, w));
		vec3 v = cross(w, u);

		origin = lookfrom;
		horizontal = viewport_width * u;
		vertical = viewport_height * v;
		lower_left_corner = origin - horizontal/2 - vertical/2 - w;
	}
	
	__device__
	ray get_ray(double s,double t)
	{
		return ray(origin,lower_left_corner + s*horizontal + t*vertical -origin);
	}

	private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
};
