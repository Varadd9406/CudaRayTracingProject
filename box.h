#pragma once
class box : public hittable 
{
    public:
		__device__
        box() {}
		__device__
		box(const point3& p0, const point3& p1, material *ptr) 
		{
			box_min = p0;
			box_max = p1;
			sides = new hittable_list(20);

			(*sides).add(new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
			(*sides).add(new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));

			(*sides).add(new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
			(*sides).add(new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));

			(*sides).add(new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
			(*sides).add(new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));
		}
		__device__
		bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override
		{
			return (*sides).hit(r, t_min, t_max, rec);
		}


    public:
        point3 box_min;
        point3 box_max;
		hittable_list *sides;
};