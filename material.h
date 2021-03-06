#pragma once
struct hit_record;

class material
{
public:
	__device__
	material() {}
	__device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, double &pdf ,curandState *thread_rand_state) const = 0;
	__device__ virtual double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const
	{
		return 1;
	}
	__device__ virtual color emitted(double u, double v, const point3 &p) const
	{
		return color(0, 0, 0);
	}
};

class lambertian : public material
{
public:
	__device__
	lambertian(textureMat *x)
	{
		albedo = x;
	}

	__device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, double &pdf,curandState *thread_rand_state) const override
	{
		onb uvw;
	    uvw.build_from_w(rec.normal);
		vec3 scatter_direction = uvw.local(random_cosine_direction(thread_rand_state));
		scattered = ray(rec.p, unit_vector(scatter_direction));
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		pdf = dot(uvw.w(),scattered.direction())/pi;
		return true;
	}

	__device__ double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const override
	{
		auto cosine = dot(rec.normal, unit_vector(scattered.direction()));
		return cosine < 0 ? 0 : cosine/pi;
	}

public:
	textureMat *albedo;
};

class metal : public material
{
public:
	__device__
	metal(const color &a, double f) : albedo(a), fuzz(fmin(1.0, f)) {}

	__device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, double& pdf, curandState *thread_rand_state) const override
	{
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(thread_rand_state));
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0);
	}

public:
	color albedo;
	double fuzz;
};

class dielectric : public material
{
public:
	__device__
	dielectric(double index_of_refraction) : ir(index_of_refraction) {}
	__device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, double& pdf, curandState *thread_rand_state) const override
	{
		attenuation = color(1.0, 1.0, 1.0);
		double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

		vec3 unit_direction = unit_vector(r_in.direction());
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0;
		vec3 direction;
		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double(thread_rand_state))
			direction = reflect(unit_direction, rec.normal);
		else
			direction = refract(unit_direction, rec.normal, refraction_ratio);

		scattered = ray(rec.p, direction);
		return true;
	}

public:
	double ir; // Index of Refraction

private:
	__device__ static double reflectance(double cosine, double ref_idx)
	{
		// Use Schlick's approximation for reflectance.
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow(double(1 - cosine), 5.0);
	}
};

class diffuse_light : public material
{
public:
	__device__
	diffuse_light(textureMat *a) : emit(a) {}
	__device__ bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered,  double& pdf, curandState *thread_rand_state) const override
	{
		return false;
	}
	__device__ color emitted(double u,double v, const point3& p) const
	{
		return emit->value(u,v,p);
	}
public:
	textureMat* emit;
};