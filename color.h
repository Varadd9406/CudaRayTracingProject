#pragma once


#include "vec3.h"

__device__ __host__
void write_color(vec3* final_out, int pos,color pixel_color)
{
	// Write the translated [0,255] value of each color component.
	vec3 temp_vec(static_cast<int>(255.999 * pixel_color.x()),static_cast<int>(255.999 * pixel_color.y()),static_cast<int>(255.999 * pixel_color.z()));
	final_out[pos] = temp_vec;
}