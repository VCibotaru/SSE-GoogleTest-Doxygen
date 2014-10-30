#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "methods.h"
#include <smmintrin.h>
#include <emmintrin.h>
#include <xmmintrin.h>
#include "gtest/gtest.h"

/**
@file main.cpp
This file contains the code used for testing the whole project
*/

///Path to image file that is used for testing
#define PATH_TO_LENNA "/Users/viktorchibotaru/Desktop/studies/3rd year/Prac/2(sse+gtest+doxy)/Lenna.bmp"

/**
@function ImagesEqual
Function checks if the elements of two matrixes of type T are equal
@param first - first Matrix
@param second - second Matrix
*/
template<typename T>
bool ImagesEqual(const Matrix<T> &first, const Matrix<T> &second) {
	if (first.n_rows != second.n_rows || first.n_cols != second.n_cols) {
		std::cout << "Sizes Differ" << std::endl;
		return false;
	}
	for (uint i = 0 ; i < first.n_rows ; ++i) {
		for (uint j = 0 ; j < first.n_cols ; ++j) {
			if (first(i, j) != second(i, j)) {
				std::cout << "(" << i << ", " << j << ")" << std::endl;
				std::cout << first(i, j) <<  " " << second(i, j) << std::endl;
				return false;
			}
		}
	}
	return true;
}
/**
@class SVMTest
This class is used for testing. It stores the grayscale image, the horizontal and vertical matrixes and the magnitudes` matrix
*/
class SVMTest {
public:
	///grayscale image
	Image gray;
	///horizontal Sobel matrix
	Image hor;
	///vertical Sobel matrix
	Image vert;
	///magnitudes` matrix
	floatImage magn;
	/**
	@function SVMTest
	This is the constructor for SVMTest objects.
	It also computes the 4 matrixes listed above
	@param image is a (@ref *BMP) pointer to image used for testing
	@param useSse is a bool that specifies whether sse  intrinsics will be used
	*/
	SVMTest(BMP *image, bool useSse): hor(image->TellHeight(), image->TellWidth()), vert(image->TellHeight(), image->TellWidth()) {
		gray = ImgToGrayscale(image);
		ApplySobel(gray, hor, vert, useSse);
		magn = GetMagnitude(hor, vert, useSse);
	}

};

/**
@function TEST(SSETest, FloatLoadUnload)
Test that checks that the norm of a const dummy vector is computed correctly using sse
*/
TEST(SSETest, FloatLoadUnload) {
	float *fPtr = new float[4];
	__m128i xInt = _mm_setr_epi16(1, 0, 3, 0, 0, 0, 4, 0);
	__m128i yInt = _mm_setr_epi16(0, 0, 4, 0, 1, 0, 3, 0);
	__m128 X = _mm_cvtepi32_ps(xInt);
	__m128 Y = _mm_cvtepi32_ps(yInt);
	X = _mm_mul_ps(X, X);
	Y = _mm_mul_ps(Y, Y);
	__m128 sum = _mm_add_ps(X, Y);
	__m128 res = _mm_sqrt_ps(sum);
	_mm_storeu_ps(fPtr, res);
	EXPECT_EQ(fPtr[0], 1.0);
	EXPECT_EQ(fPtr[1], 5.0);
	EXPECT_EQ(fPtr[2], 1.0);
	EXPECT_EQ(fPtr[3], 5.0);
	delete fPtr;

}
/**
@function TEST(SSETest, TestSobel)
Test that checks that horizontal and vertical Sobel matrixes are computed correctly using sse.
Firstly, it computes the matrixes without sse
Secondly, it computes them using sse
Thirdly, it calls (@ref ImagesEqual) to check whether the matrixes are equal
*/

TEST(SSETest, TestSobel) {
	BMP* image = new BMP();
	image->ReadFromFile(PATH_TO_LENNA);
	SVMTest first(image, false);
	SVMTest second(image, true);
	EXPECT_TRUE(ImagesEqual(first.hor, second.hor));
	EXPECT_TRUE(ImagesEqual(first.vert, second.vert));
	delete image;
}

/**
@function TEST(SSETest, TestMagnitude)
Test that checks that magnitudes` matrixes are computed correctly using sse.
The approach is the same as for (@ref TEST(SSETest, TestSobel))
*/

TEST(SSETest, TestMagnitude) {
	BMP* image = new BMP();
	image->ReadFromFile(PATH_TO_LENNA);
	SVMTest first(image, false);
	SVMTest second(image, true);
	EXPECT_TRUE(ImagesEqual(first.magn, second.magn));
	delete image;
}

/**
@function TEST(SSETest, TestGetDescriptor)
Test that checks that HOG descriptors are computed correctly using sse.
The approach is the same as for (@ref TEST(SSETest, TestSobel))
*/

TEST(SSETest, TestGetDescriptor) {
	BMP* image = new BMP();
	image->ReadFromFile(PATH_TO_LENNA);
	SVMTest first(image, false);
	SVMTest second(image, true);
	std::vector<float> firstResult;
	std::vector<float> secondResult;
	GetDescriptor(first.hor, first.vert, first.magn, firstResult);
	GetDescriptor(second.hor, second.vert, second.magn, secondResult);
	EXPECT_EQ(firstResult, secondResult);
	delete image;
}



/**
@function main
Runs all tests
*/
int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}