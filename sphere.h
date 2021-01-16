#pragma once




class sphere :public hittable
{
	public:
	__device__ sphere() {}
	__device__ sphere(point3 cen, double r) : center(cen), radius(r) {}
    __device__ virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;


	public:
	point3 center;
	double radius;
};


__device__ bool sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
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
	return true;
}