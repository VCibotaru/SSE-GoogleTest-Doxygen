#ifndef METHODS_H_
#define METHODS_H_


#include "matrix.h"
#include "EasyBMP.h"



const uint CELL_COUNT  = 16;
const uint SEGMENT_COUNT = 16;
const float L = 0.5;
const int N = 1;
const int COLOR_CELL_COUNT = 8;
const uint FILTER_RADIUS = 1;
const uint SSE_BLOCK_SIZE = 8;
const uint SSE_FLOAT_BLOCK_SIZE = 4;

typedef Matrix<short> Image;
typedef Matrix<float> floatImage;

class VertSobel 
{
public:
    const int vert_radius;
    const int hor_radius;
    VertSobel() : vert_radius(1), hor_radius(1) {}
    short operator () (const Image &mat) const
    {
        return -mat(0, 0) - 2 * mat(0, 1) - mat(0, 2) + mat(2, 0) + 2 * mat(2, 1) + mat(2, 2);
    }
};

class HorSobel
{
public:
    const int vert_radius;
    const int hor_radius;
    HorSobel() : vert_radius(1), hor_radius(1) {}
    short operator () (const Image &mat) const
    {
        return -mat(0, 0) - 2 * mat(1, 0) - mat(2, 0) + mat(0, 2) + 2 * mat(1, 2) + mat(2, 2);
    }
};

Image ImgToGrayscale(BMP *img);
floatImage GetMagnitude(const Image &hor, const Image &vert, bool useSse);
void ApplySobel(const Image &img, Image &hor, Image &vert, bool useSse);
void GetDescriptor(const Image &hor, const Image &vert, const floatImage &magn, std::vector<float> &result);
void GetColors(BMP *img, std::vector<float> &result);
std::vector<float> GetHist(const Image &hor, const Image &vert, const floatImage &magn);
std::vector<float> ApplyHIKernel(const std::vector<float> &preHI);

#endif