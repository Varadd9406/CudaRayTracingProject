class xy_rect : public hittable
{
public:
	__device__
	xy_rect() {}
	__device__
	xy_rect(double _x0, double _x1, double _y0, double _y1, double _k, material *mat)
		: x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mat_ptr(mat){};
	__device__ bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;

public:
	material *mat_ptr;
	double x0, x1, y0, y1, k;
};

class xz_rect : public hittable
{
public:
	__device__
	xz_rect() {}
	__device__
	xz_rect(double _x0, double _x1, double _z0, double _z1, double _k,
			material *mat)
		: x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat){};
	__device__ bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;

public:
	material *mat_ptr;
	double x0, x1, z0, z1, k;
};

class yz_rect : public hittable
{
public:
	__device__
	yz_rect() {}
	__device__
	yz_rect(double _y0, double _y1, double _z0, double _z1, double _k,
			material *mat)
		: y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mat_ptr(mat){};
	__device__ virtual bool hit(const ray &r, double t_min, double t_max, hit_record &rec) const override;

public:
	material *mat_ptr;
	double y0, y1, z0, z1, k;
};

__device__ bool xy_rect::hit(const ray &r, double t_min, double t_max, hit_record &rec) const
{
	auto t = (k - r.origin().z()) / r.direction().z();
	if (t < t_min || t > t_max)
		return false;
	auto x = r.origin().x() + t * r.direction().x();
	auto y = r.origin().y() + t * r.direction().y();
	if (x < x0 || x > x1 || y < y0 || y > y1)
		return false;
	rec.u = (x - x0) / (x1 - x0);
	rec.v = (y - y0) / (y1 - y0);
	rec.t = t;
	auto outward_normal = vec3(0, 0, 1);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;
	rec.p = r.at(t);
	return true;
}
__device__ bool xz_rect::hit(const ray &r, double t_min, double t_max, hit_record &rec) const
{
	auto t = (k - r.origin().y()) / r.direction().y();
	if (t < t_min || t > t_max)
		return false;
	auto x = r.origin().x() + t * r.direction().x();
	auto z = r.origin().z() + t * r.direction().z();
	if (x < x0 || x > x1 || z < z0 || z > z1)
		return false;
	rec.u = (x - x0) / (x1 - x0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	auto outward_normal = vec3(0, 1, 0);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;
	rec.p = r.at(t);
	return true;
}

__device__ bool yz_rect::hit(const ray &r, double t_min, double t_max, hit_record &rec) const
{
	auto t = (k - r.origin().x()) / r.direction().x();
	if (t < t_min || t > t_max)
		return false;
	auto y = r.origin().y() + t * r.direction().y();
	auto z = r.origin().z() + t * r.direction().z();
	if (y < y0 || y > y1 || z < z0 || z > z1)
		return false;
	rec.u = (y - y0) / (y1 - y0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	auto outward_normal = vec3(1, 0, 0);
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;
	rec.p = r.at(t);
	return true;
}