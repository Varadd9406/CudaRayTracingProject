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
vec3 random_in_unit_sphere(curandState *thread_rand_state)
{
    while (true)
	{
        auto p = random_vec3(thread_rand_state,-1,1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}