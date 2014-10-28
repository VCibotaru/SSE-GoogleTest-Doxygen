#include "methods.h"
#include "EasyBMP.h"
#include <math.h>

#define R 0.299
#define G 0.587
#define B 0.114

Image ImgToGrayscale(BMP *img) {
	Image newImg(img->TellHeight(), img->TellWidth());
	for (uint i = 0 ; i < newImg.n_rows ; ++i) {
		for (uint j = 0 ; j < newImg.n_cols ; ++j) {
			auto pixel = img->GetPixel(j, i);
			int value = pixel.Red * R + pixel.Green * G + pixel.Blue * B;
			newImg(i, j) = value;
		}
	}
	return newImg;
}

std::vector<float> GetHist(const Image &hor, const Image &vert) {
	std::vector<float> result(SEGMENT_COUNT);
	for (uint i = 0 ; i < hor.n_rows ; ++i) {
		for (uint j = 0 ; j < hor.n_cols ; ++j) {
			float angle = atan2(vert(i,j), hor(i,j));
			uint section = uint(SEGMENT_COUNT * (angle + M_PI) / (2 * M_PI));
			section = (section == SEGMENT_COUNT) ? SEGMENT_COUNT - 1 : section;
			result[section] += sqrt(pow(hor(i, j), 2) + pow(vert(i, j), 2) );
		}
	}
	float sum = 0;
	for (uint i = 0 ; i < SEGMENT_COUNT ; ++i) {
		sum += result[i] * result[i];
	}
	sum = sqrt(sum);
	if (sum) {
		for (uint i = 0 ; i < SEGMENT_COUNT ; ++i) {
			result[i] /= sum;
		}
	}
	return result;
}

float sech(float x) {
	return 2.0 / (exp(x) + exp(-x));
}
std::vector<float> ApplyHIKernel(const std::vector<float> &preHI) {
	std::vector <float> postHI;
	for (float x : preHI) {
		for (int j = -N ; j <= N ; ++j) {
			if (x) {
				float im = - sqrt(x * sech(M_PI * j * L)) * sin(j * L * log(x));
				float re = sqrt(x * sech(M_PI * j * L)) * cos(j * L * log(x));
				postHI.push_back(im);
				postHI.push_back(re);
			}
			else {
				postHI.push_back(0.0);
				postHI.push_back(0.0);
			}
		}
	}
	return postHI;
}

void GetDescriptor(const Image &hor, const Image &vert, std::vector<float> &result) {
	for (uint i = 0 ; i < CELL_COUNT ; ++i) {
            for (uint j = 0 ; j < CELL_COUNT ; ++j) {
                uint rows = (i == CELL_COUNT - 1) ? hor.n_rows - i * hor.n_rows / CELL_COUNT : hor.n_rows / CELL_COUNT;
                uint cols = (j == CELL_COUNT - 1) ? hor.n_cols - j * hor.n_cols / CELL_COUNT : hor.n_cols / CELL_COUNT;
                uint x = i * hor.n_rows / CELL_COUNT;
                uint y = j * hor.n_cols / CELL_COUNT;
                Image subHor = hor.submatrix(x, y, rows, cols);
                Image subVert = vert.submatrix(x, y, rows, cols);
                std::vector<float> tmp = GetHist(subHor, subVert);
                result.insert(result.end(), tmp.begin(), tmp.end());
            } 
        }
}

void GetCellColors(BMP *img, std::vector<float> &result, uint rows, uint cols, uint x, uint y) {
	uint sumR, sumB, sumG, count = rows * cols;
	sumR = sumB = sumG = 0;
	for (uint i = x ; i < x + rows ; ++i) {
		for (uint j = y ; j < y + cols ; ++j) {
			auto pixel = img->GetPixel(y, x);
			sumR += pixel.Red;
			sumB += pixel.Blue;
			sumG += pixel.Green;
		}
	}
	result.push_back(float(sumR) / (count * 255.0));
	result.push_back(float(sumG) / (count * 255.0));
	result.push_back(float(sumB) / (count * 255.0));

}

void GetColors(BMP *img, std::vector<float> &result) {
	for (int i = 0 ; i < COLOR_CELL_COUNT ; ++i) {
            for (int j = 0 ; j < COLOR_CELL_COUNT ; ++j) {
                uint rows = (i == COLOR_CELL_COUNT - 1) ? img->TellHeight() - i * img->TellHeight() / COLOR_CELL_COUNT : img->TellHeight() / COLOR_CELL_COUNT;
                uint cols = (j == COLOR_CELL_COUNT - 1) ? img->TellWidth() - j * img->TellWidth() / COLOR_CELL_COUNT : img->TellWidth() / COLOR_CELL_COUNT;
                uint x = i * img->TellHeight() / COLOR_CELL_COUNT;
                uint y = j * img->TellWidth() / COLOR_CELL_COUNT;
                GetCellColors(img, result, rows, cols, x, y);
            }
        }
}