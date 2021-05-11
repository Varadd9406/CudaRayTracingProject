class raw_img
{
	public:
	__host__ __device__
	raw_img(unsigned char* data_ptr,int p_width,int p_height)
	{
		data = data_ptr;
		width = p_width;
		height = p_height;
	}


	public:
	unsigned char* data;
	int width;
	int height;
};