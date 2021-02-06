#pragma once

class textureMat
{
	public:
	__device__
	virtual color value(double u,double v,const point3& p) const = 0;
};

class solid_color : public textureMat
{
	public:
	__device__
	solid_color() {}
	
	__device__
	solid_color(color c):color_value(c) {}

	__device__
	color value(double u, double v, const vec3& p) const override
	{
		return color_value;
	}
	private:
	color color_value;
};