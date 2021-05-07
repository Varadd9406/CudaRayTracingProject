#pragma once



__device__
void get_moving_sphere_screen_coordinates(const point3 &p,double &u,double &v);

class moving_sphere :public hittable
{
	public:
	__device__ moving_sphere() {}
	__device__ moving_sphere(path* p, double r,material* m) : path_ptr(p), radius(r),mat_ptr(m) 
	{
		center = (*path_ptr).position();
	}
    __device__ bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const override;

	__device__ void move()
	{
		center = (*path_ptr).position();
	}

	__device__ ~moving_sphere()
	{
		delete mat_ptr;
	}


	public:
	point3 center;
	path* path_ptr;
	double radius;
	material* mat_ptr;
};


__device__ 
bool moving_sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
	vec3 oc = r.origin() - center;
	double a = r.direction().length_squared();
	double half_b = dot(oc, r.direction());
	double c = oc.length_squared() - radius*radius;

	double discriminant = half_b*half_b - a*c;
	if (discriminant < 0) return false;
	double sqrtd = sqrt(discriminant);

	// Find the nearest root that lies in the acceptable range.
	double root = (-half_b - sqrtd) / a;
	if (root < t_min || t_max < root)
	{
		root = (-half_b + sqrtd) / a;
		if (root < t_min || t_max < root)
			return false;
	}

	rec.t = root;
	rec.p = r.at(rec.t);
	vec3 outward_normal = (rec.p - center) / radius;
	rec.set_face_normal(r, outward_normal);
	get_moving_sphere_screen_coordinates(outward_normal,rec.u,rec.v);
	rec.mat_ptr = mat_ptr;
	return true;
}



__device__
void get_moving_sphere_screen_coordinates(const point3 &p,double &u,double &v)
{
	double theta = acos(-p.y());
	double phi = atan2(-p.z(),p.x()) + pi;

	u = phi/(2*pi);
	v = theta/pi; 
}