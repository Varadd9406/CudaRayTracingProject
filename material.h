#ifndef MATERIALH
#define MATERIALH
struct hit_record;


#include"hittable.h"


class material
{
	public:
	__device__
	material(){}
	__device__
	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *thread_rand_state) const =0;
	
};

class lambertian: public material
{
	public:
	__device__
	lambertian(const color &a):albedo(a) {}

	__device__
	bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState *thread_rand_state) const override 
	{
		vec3 scatter_direction = rec.normal + random_unit_vector(thread_rand_state);
		if (scatter_direction.near_zero())
		{
        	scatter_direction = rec.normal;
		}
		scattered = ray(rec.p, scatter_direction);
		attenuation = albedo;
		return true;
    }
	public:
	color albedo;
	
};


class metal : public material 
{
    public:
		__device__
        metal(const color& a,double f) : albedo(a), fuzz(fmin(1.0,f)) {}

		__device__
        bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered,curandState *thread_rand_state) const override
		{
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected+fuzz*random_in_unit_sphere(thread_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

    public:
        color albedo;
		double fuzz;
};
#endif
