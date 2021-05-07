#pragma once


class circular_path : public path
{
	public:
	__device__
	circular_path() {}
	__device__
	circular_path(point3 p_cen,point3 p_u,point3 p_v,double p_radius,int p_rev,int target_fps,double running_time)
	{
		cen = p_cen;
		u = unit_vector(p_u - p_cen);
		v = unit_vector(p_v -p_cen);
		radius = p_radius;
		curr_angle = 0;
		step_per_frame = double(2*pi*p_rev)/double(running_time*target_fps);
		curr_pos = cen + p_radius*cos(curr_angle)*u + p_radius*sin(curr_angle)*v ;

	}
	__device__ 
	vec3 position() override
	{
		point3 ans = curr_pos;
		curr_angle += step_per_frame ;
		curr_pos = cen + radius*cos(curr_angle)*u +radius*sin(curr_angle)*v ;
		return ans;
	}
	public:
	point3 cen,u,v,curr_pos;
	double radius,curr_angle,step_per_frame;

};