#ifndef HITTABLEH
#define HITTABLEH


class material;
struct hit_record
{
	__device__
	void set_face_normal(const ray& r, const vec3& outward_normal)
	{
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
	material* mat_ptr;
	point3 p;
	vec3 normal;
	double t;
	bool front_face;
};

class hittable
{
	public:
	__device__ __host__
	hittable() {}
	__device__ 
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;

};

#endif
