#pragma once


class hittable_list :public hittable
{
	public:
	__device__
	hittable_list()
	{
		// m_cnt =0;
		// cudaMalloc((void **)&objects,10*sizeof(hittable *));
	}

	__device__
	hittable_list(int mx)
	{
		objects =  new hittable*[mx];
		cnt = 0;
		max_elem = mx;
	}

	__device__
	void add(hittable* obj)
	{
		objects[cnt++] = obj;
	}

    __device__ bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const override;


	__device__
	~hittable_list()
	{
		for (int i = 0; i < cnt; i++)
		{
			delete objects[i];
		}
	}

	public:
	int cnt;
	int max_elem;
	hittable** objects;
};

__device__
bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
	hit_record temp_rec;
	bool hit_anything = false;
	auto closest_so_far = t_max;


	for (int i =0;i<cnt;i++)
	{
		if ((*objects[i]).hit(r, t_min, closest_so_far, temp_rec))
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}