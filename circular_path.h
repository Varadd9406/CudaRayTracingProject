#pragma once


class circular_path : public path
{
	public:
	__device__
	circular_path() {}
	__device__
	circular_path(point3 p_cen,point3 p_u,point3 p_v,double p_radius,double p_rev_per_sec,int target_fps, double start_angle = 0)
	{
		cen = new stationary_path(p_cen);
		u = unit_vector(p_u - p_cen);
		v = unit_vector(p_v -p_cen);
		radius = p_radius;
		curr_angle = start_angle;
		step_per_frame = double(2*pi*p_rev_per_sec)/double(target_fps);
		curr_pos = cen->position() + p_radius*cos(curr_angle)*u + p_radius*sin(curr_angle)*v ;
	}
	__device__
	circular_path(path* p_cen,point3 p_u,point3 p_v,double p_radius,double p_rev_per_sec,int target_fps, double start_angle = 0)
	{
		cen = p_cen;
		curr_cen_pos = p_cen->position() ;
		u = p_u;
		v = p_v;
		radius = p_radius;
		curr_angle = start_angle;
		step_per_frame = double(2*pi*p_rev_per_sec)/double(target_fps);
		curr_pos = curr_cen_pos + p_radius*cos(curr_angle)*unit_vector(u -curr_cen_pos) + p_radius*sin(curr_angle)*unit_vector(v -curr_cen_pos) ;
	}
	__device__ 
	vec3 position() override
	{
		point3 ans = curr_pos;
		curr_angle += step_per_frame ;
		curr_cen_pos = cen->position();
		curr_pos = curr_cen_pos + radius*cos(curr_angle)*unit_vector(u -curr_cen_pos) +radius*sin(curr_angle)*unit_vector(v -curr_cen_pos) ;
		return ans;
	}
	public:
	path* cen;
	point3 u,v,curr_pos,curr_cen_pos;
	double radius,curr_angle,step_per_frame;

};