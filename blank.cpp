#include <iostream>


class A
{
	public:
	virtual void hit() = 0;
};
class B : public A
{
	public:
	void hit() override
	{
		std::cout<<"Im HIT";
	}
};



void func(A* ptr)
{
	(*ptr).hit();
}

int main()
{
	B b;
	func(&b);
}