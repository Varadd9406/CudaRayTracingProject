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

class image_texture : public textureMat 
{
    public:
		__device__
		image_texture(): data(nullptr), width(0), height(0) {}
		__device__
        image_texture(unsigned char * p_data,int p_width,int p_height):data(p_data),width(p_width),height(p_height) {}

		__device__
        color value(double u, double v, const vec3& p) const override {
            // If we have no texture data, then return solid cyan as a debugging aid.
            if (data == nullptr)
                return color(0,1,1);

            // Clamp input texture coordinates to [0,1] x [1,0]
            u = clamp(u, 0.0, 1.0);
            v = 1.0 - clamp(v, 0.0, 1.0);  // Flip V to image coordinates

            auto i = static_cast<int>(u * width);
            auto j = static_cast<int>(v * height);

            // Clamp integer mapping, since actual coordinates should be less than 1.0
            if (i >= width)  i = width-1;
            if (j >= height) j = height-1;

            const auto color_scale = 1.0 / 255.0;
            auto r = *(data + j*width + i);
			auto g = *(data + j*width + i + width*height);
			auto b = *(data + j*width + i + 2*width*height);

            return color(color_scale*r, color_scale*g, color_scale*b);
        }

    private:
        unsigned char *data;
        int width, height;
};