#pragma once


#include <math.h>
#include <iostream>


class vec3
{
    public:
		__device__ __host__
        vec3() : e{0,0,0} {}
		__device__ __host__
        vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}
		__device__ __host__
        double x() const { return e[0]; }
		__device__ __host__
        double y() const { return e[1]; }
		__device__ __host__
        double z() const { return e[2]; }
		__device__ __host__
        vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
		__device__ __host__
        double operator[](int i) const { return e[i]; }
		__device__ __host__
        double& operator[](int i) { return e[i]; }
		__device__ __host__
        vec3& operator+=(const vec3 &v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }
		__device__ __host__
        vec3& operator*=(const double t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }
		__device__ __host__
        vec3& operator*=(const vec3 &v) {
            e[0] *= v.e[0];
            e[1] *= v.e[1];
            e[2] *= v.e[2];
            return *this;
        }
		__device__ __host__
        vec3& operator/=(const double t) {
            return *this *= 1/t;
        }
		__device__ __host__
        double length() const {
            return sqrt(length_squared());
        }
		__device__ __host__
        double length_squared() const {
            return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
        }
		__device__ __host__
    	bool near_zero() const 
		{
			const auto s = 1e-8;
        	return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    	}

    public:
        double e[3];
};

// Type aliases for vec3
using point3 = vec3;   // 3D point
using color = vec3;    // RGB color


std::ostream& operator<<(std::ostream &out, const vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}
__device__ __host__
vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}
__device__ __host__
vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}
__device__ __host__
vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}
__device__ __host__
vec3 operator*(double t, const vec3 &v) {
    return vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}
__device__ __host__
vec3 operator*(const vec3 &v, double t) {
    return t * v;
}
__device__ __host__
vec3 operator/(vec3 v, double t) {
    return (1/t) * v;
}
__device__ __host__
double dot(const vec3 &u, const vec3 &v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}
__device__ __host__
vec3 cross(const vec3 &u, const vec3 &v){
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}
__device__ __host__
vec3 unit_vector(vec3 v){
    return v / v.length();
}

__device__ 
vec3 random_vec3(curandState *thread_rand_state)
{
    return vec3(random_double(thread_rand_state), random_double(thread_rand_state), random_double(thread_rand_state));
}

__device__
vec3 random_vec3(curandState *thread_rand_state,double min, double max)
{
    return vec3(random_double(thread_rand_state,min,max), random_double(thread_rand_state,min,max), random_double(thread_rand_state,min,max));
}

__device__
vec3 random_in_unit_sphere(curandState *rand_state)
{
    while (true)
	{
        auto p = random_vec3(rand_state,-1,1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

__device__
vec3 random_unit_vector(curandState *rand_state)
{
    return unit_vector(random_in_unit_sphere(rand_state));
}

__device__
vec3 reflect(const vec3& v, const vec3& n) 
{
    return v - 2*dot(v,n)*n;
}

__device__
vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat)
 {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}