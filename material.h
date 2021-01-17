#pragma once

class material
{
	public:
	__device__
	material(){}
	__device__
	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const =0;
	
};

class lambertian: public material
{
	public:
	__device__
	lambertian(const color&a):albedo(a) {}
	
}
