#pragma once


#include "hittable.h"
#include "vec3.h"


class sphere :public hittable
{
	public:
	__device__ __host__
	sphere(){}

	__device__ __host__
	sphere(point3 cen, double r) : center(cen), radius(r) {}
	
	__device__ __host__
	hit(const ray& r, double t_min, double t_max, hit_record& rec)
	{
		vec3 oc = r.origin() - center;
		auto a = r.direction().length_squared();
		auto half_b = dot(oc, r.direction());
		auto c = oc.length_squared() - radius*radius;

		auto discriminant = half_b*half_b - a*c;
		if (discriminant < 0) return false;
		auto sqrtd = sqrt(discriminant);

		// Find the nearest root that lies in the acceptable range.
		auto root = (-half_b - sqrtd) / a;
		if (root < t_min || t_max < root)
		{
			root = (-half_b + sqrtd) / a;
			if (root < t_min || t_max < root)
				return false;
		}

		rec.t = root;
		rec.p = r.at(rec.t);
		rec.normal = (rec.p - center) / radius;

		return true;
	}
	public:
	point3 center;
	double radius;
};
