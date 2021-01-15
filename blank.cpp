#include <iostream>

int main()
{
	#ifndef ONLINE_JUDGE
		freopen("image2.ppm", "w", stdout);
	#endif

    // Image

    const int image_width = 1920;
    const int image_height = 1080;

    // Render

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = image_height-1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            auto r = float(i) / float(image_width-1);
            auto g = float(j) / float(image_height-1);
            auto b = 0.25;

            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);

            // std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
}