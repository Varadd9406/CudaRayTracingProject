#pragma once


#include "vec3.h"

class ray {
    public:
		__device__ __host__
        ray() {}

		__device__ __host__
        ray(const point3& origin, const vec3& direction)
            : orig(origin), dir(direction)
        {}

		__device__ __host__
        point3 origin() const  { return orig; }
		__device__ __host__
        vec3 direction() const { return dir; }
		__device__ __host__
        point3 at(double t) const {
            return orig + t*dir;
        }

    public:
        point3 orig;
        vec3 dir;
};

