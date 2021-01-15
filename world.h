#pragma once

#include "hittable.h"

class world
{
	public:
	__device__ __cuda__
	world(){}
	
	void add(hittable *object)
	{
		objects[m_cnt++] = object;
	}

	__device__ __host__
	bool hit(const ray& r, double t_min, double t_max, hit_record& rec)
	{
		hit_record temp_rec;
		bool hit_anything = false;
		auto closest_so_far = t_max;

		for (int i =0;i<m_cnt;i++)
		{
			if ((*object)->hit(r, t_min, closest_so_far, temp_rec))
			{
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
    	return hit_anything;
	}




	public:
	int m_cnt=0;
	hittable* objects[100];
	int max_capacity = 100;
}