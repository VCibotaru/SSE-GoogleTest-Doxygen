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
#include "gtest/gtest.h"

#define PATH_TO_LENNA "/Users/viktorchibotaru/Desktop/studies/3rd year/Prac/2(sse+gtest+doxy)/Lenna.bmp"

bool ImagesEqual(const Image &first, const Image &second) {
	if (first.n_rows != second.n_rows || first.n_cols != second.n_cols) {
		return false;
	}
	for (uint i = 0 ; i < first.n_rows ; ++i) {
		for (uint j = 0 ; j < first.n_cols ; ++j) {
			if (first(i, j) != second(i, j)) {
				return false;
			}
		}
	}
	return true;
}

class SVM {
public:
	Image gray;
	Image hor;
	Image vert;	
	SVM(BMP *image): hor(image->TellHeight(), image->TellWidth()), vert(image->TellHeight(), image->TellWidth()) {
		gray = ImgToGrayscale(image);

	}
	void Sobel(bool useSse) {
		ApplySobel(gray, hor, vert, useSse);
	}

};

TEST(SSETest, TestSobel) {
	BMP* image = new BMP();
	image->ReadFromFile(PATH_TO_LENNA);
	SVM first(image);
	SVM second(image);
	first.Sobel(false);
	second.Sobel(true);
	EXPECT_TRUE(ImagesEqual(first.hor, second.hor));
	EXPECT_TRUE(ImagesEqual(first.vert, second.vert));
	delete image;
}

TEST(SSETest, TestMagnitude) {
	BMP* image = new BMP();
	image->ReadFromFile(PATH_TO_LENNA);
	SVM first(image);
	SVM second(image);
	first.Sobel(false);
	second.Sobel(true);
	std::vector<float> firstResult;
	std::vector<float> secondResult;
	firstResult = GetHist(first.hor, first.vert, false);
	secondResult = GetHist(first.hor, first.vert, true);
	EXPECT_EQ(firstResult, secondResult);
	delete image;
}

TEST(SSETest, TestGetDescriptor) {
	BMP* image = new BMP();
	image->ReadFromFile(PATH_TO_LENNA);
	SVM first(image);
	SVM second(image);
	first.Sobel(false);
	second.Sobel(true);
	std::vector<float> firstResult;
	std::vector<float> secondResult;
	GetDescriptor(first.hor, first.vert, firstResult, false);
	GetDescriptor(first.hor, first.vert, secondResult, true);
	EXPECT_EQ(firstResult, secondResult);
	delete image;
}




int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}