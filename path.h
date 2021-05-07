#pragma once


class path
{
	public:
	__device__
	path() {}
	__device__
	virtual vec3 position() = 0;

};