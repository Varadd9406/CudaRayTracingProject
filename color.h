#pragma once



__device__
void write_color(vec3* final_out, int pos,color pixel_color,int sample_size)
{
	// Write the translated [0,255] value of each color component.
	double r = pixel_color.x();
	double g = pixel_color.y();
	double b = pixel_color.z();
	double scale =1.0/double(sample_size);
	r=sqrt(r*scale);
	g=sqrt(g*scale);
	b=sqrt(b*scale);
	

	vec3 temp_vec(static_cast<int>(255.999 *clamp(r,0.0,0.999)),static_cast<int>(255.999 *clamp(g,0.0,0.999)),static_cast<int>(255.999 *clamp(b,0.0,0.999)));
	final_out[pos] = temp_vec;
}