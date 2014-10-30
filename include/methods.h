#ifndef METHODS_H_
#define METHODS_H_


#include "matrix.h"
#include "EasyBMP.h"

/**
@file methods.h
The header that contains the description of main functions used for computing HOG descriptor
*/

///Specifies in how many cells (both vertical and horizontal) the image will be divided when computing HOG
const uint CELL_COUNT  = 16;
///Specifies in how many subsegments will be divided [-pi, pi] when computing HOG
const uint SEGMENT_COUNT = 16;
///The L constant used for non-line kernel
const float L = 0.5;
///The N constant used for non-line kernel
const int N = 1;
///Specifies in how many cells (both vertical and horizontal) the image will be divided when computing color features
const int COLOR_CELL_COUNT = 8;
///Specifies the radius of Sobel filters
const uint FILTER_RADIUS = 1;
///Specifies how many shorts will be packed in __m128i
const uint SSE_BLOCK_SIZE = 8;
//////Specifies how many floats will be packed in __m128
const uint SSE_FLOAT_BLOCK_SIZE = 4;

///Matrix of shorts
typedef Matrix<short> Image;
///Matrix of floats
typedef Matrix<float> floatImage;

/**
@class VertSobel
Class used for Matrix::unary_map that computes the vertical Sobel matrix
*/
class VertSobel 
{
public:
    ///Specifies the vertical radius of the filter
    const int vert_radius;
    ///Specifies the horizontal radius of the filter
    const int hor_radius;
    VertSobel() : vert_radius(FILTER_RADIUS), hor_radius(FILTER_RADIUS) {}
    ///Operator that computes the vertical Sobel matrix for a (2 * (@ref hor_radius) + 1) x (2 * (@ref vert_radius) + 1) submatrix
    short operator () (const Image &mat) const
    {
        return -mat(0, 0) - 2 * mat(0, 1) - mat(0, 2) + mat(2, 0) + 2 * mat(2, 1) + mat(2, 2);
    }
};

/**
@class HorSobel
Class used for Matrix::unary_map that computes the horizontal Sobel matrix
*/
class HorSobel
{
public:
    ///Specifies the vertical radius of the filter
    const int vert_radius;
    ///Specifies the horizontal radius of the filter
    const int hor_radius;
    HorSobel() : vert_radius(FILTER_RADIUS), hor_radius(FILTER_RADIUS) {}
    ///Operator that computes the horizontal Sobel matrix for a (2 * (@ref hor_radius) + 1) x (2 * (@ref vert_radius) + 1) submatrix
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