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

class checker_texture : public textureMat {
    public:
		__device__
        checker_texture() {}
		__device__
        checker_texture(textureMat* _even, textureMat* _odd) : even(_even), odd(_odd) {}

		__device__
        color value(double u, double v, const point3& p) const override 
		{
            double sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
            if (sines < 0)
                return odd->value(u, v, p);
            else
                return even->value(u, v, p);
        }

    public:
        textureMat* odd;
        textureMat* even;
};