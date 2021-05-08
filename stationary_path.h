#pragma once


class stationary_path : public path
{
	public:
	__device__
	stationary_path() {}
	__device__
	stationary_path(point3 p_cen)
	{
		cen = p_cen;
	}
	__device__ 
	vec3 position() override
	{
		return cen;
	}
	public:
	point3 cen;

};