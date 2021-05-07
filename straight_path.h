#pragma once


class straight_path : public path
{
	public:
	__device__
	straight_path() {}
	__device__
	straight_path(point3 p_orig,point3 p_dir,double p_time1,double p_time2,int target_fps,double running_time)
	{
		orig = p_orig;
		dir = unit_vector(p_dir);
		time1 = p_time1;
		time2 = p_time2;
		step_per_frame = double(time2-time1)/double(running_time*target_fps);
		curr_pos = orig + time1*dir;

	}
	__device__ 
	vec3 position() override
	{
		point3 ans = curr_pos;
		curr_pos += step_per_frame*dir;
		return ans;
	}
	public:
	point3 orig,dir,curr_pos;
	double time1,time2,step_per_frame;

};