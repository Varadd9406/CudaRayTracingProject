#include "ray.h"
class hit_record
{
	public:
	__device__ __cuda__
	hit_record(){}

	__device__ __cuda__
	void set_face_normal(const ray& r, const vec3& outward_normal)
	{
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
	public:
	point3 p;
	vec3 normal;
	double t;
	bool front_face;
};

class hittable
{
	public:
	__device__ __cuda__
	hittable(){}
};


