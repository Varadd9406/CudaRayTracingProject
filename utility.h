#pragma once

#include <math.h>


// Constants

#define infinity INT32_MAX
#define pi 3.1415926535897932385

// Utility Functions


__device__
double degrees_to_radians(double degrees)
{
	return (degrees*pi)/180.0;
}


__device__
double clamp(double x,double mn,double mx)
{
	if (x<mn) return mn;
	if (x>mx) return mx;
	return x;
}
// Common Headers

