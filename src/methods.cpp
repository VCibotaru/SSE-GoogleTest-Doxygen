#include "methods.h"
#include "EasyBMP.h"
#include <smmintrin.h>
#include <emmintrin.h>
#include <math.h>

#define RED 0.299
#define GREEN 0.587
#define BLUE 0.114

Image ImgToGrayscale(BMP *img) {
	Image newImg(img->TellHeight(), img->TellWidth());
	for (uint i = 0 ; i < newImg.n_rows ; ++i) {
		for (uint j = 0 ; j < newImg.n_cols ; ++j) {
			auto pixel = img->GetPixel(j, i);
			short value = pixel.Red * RED + pixel.Green * GREEN + pixel.Blue * BLUE;
			newImg(i, j) = value;
		}
	}
	return newImg;
}

void ApplySobel(const Image &img, Image &hor, Image &vert, bool useSse) {
	if (!useSse) {
		hor = img.unary_map(HorSobel());
		vert = img.unary_map(VertSobel());
	}
	else {
		Image extraImg = img.extra_borders(FILTER_RADIUS, FILTER_RADIUS);
		uint j, leftElems = img.n_cols % SSE_BLOCK_SIZE;
		uint blockElems =  img.n_cols - leftElems;
		uint stride = extraImg.getStride();
		short *ptr = extraImg.getData().get();
		short *horPtr = hor.getData().get();
		short *vertPtr = vert.getData().get();
		for (uint i = 0 ; i < img.n_rows ; ++i) {
			for (j = 0; j <  blockElems ; j += SSE_BLOCK_SIZE) {
				__m128i A = _mm_loadu_si128((__m128i *) (ptr + i       * stride + j    ));
				__m128i B = _mm_loadu_si128((__m128i *) (ptr + i       * stride + j + 1));
				__m128i C = _mm_loadu_si128((__m128i *) (ptr + i       * stride + j + 2));
				__m128i D = _mm_loadu_si128((__m128i *) (ptr + (i + 1) * stride + j    ));
				__m128i F = _mm_loadu_si128((__m128i *) (ptr + (i + 1) * stride + j + 2));
				__m128i G = _mm_loadu_si128((__m128i *) (ptr + (i + 2) * stride + j    ));
				__m128i H = _mm_loadu_si128((__m128i *) (ptr + (i + 2) * stride + j + 1));
				__m128i I = _mm_loadu_si128((__m128i *) (ptr + (i + 2) * stride + j + 2));
					//X = (D - F) + (D - F) + A - I - C + G
					//Y = (B - H) + (B - H) + A - I + C - G


				__m128i  tmpX = _mm_sub_epi16(F, D);
				__m128i  tmpY = _mm_sub_epi16(H, B);
				__m128i  tmpAI = _mm_sub_epi16(I, A);
				__m128i  tmpCG = _mm_sub_epi16(C, G);

				__m128i X = _mm_add_epi16(tmpX, tmpX);
				X = _mm_add_epi16(X, tmpAI);
				X = _mm_add_epi16(X, tmpCG);

				__m128i Y = _mm_add_epi16(tmpY, tmpY);
				Y = _mm_add_epi16(Y, tmpAI);
				Y = _mm_sub_epi16(Y, tmpCG);

				_mm_storeu_si128((__m128i *) (horPtr +  i * img.getStride() + j), X);
				_mm_storeu_si128((__m128i *) (vertPtr + i * img.getStride() + j), Y);

			}
			for (; j < img.n_cols ; ++j) {
				auto mat = extraImg.submatrix(i, j, 2 * FILTER_RADIUS + 1, 2 * FILTER_RADIUS + 1);
				hor(i, j) = -mat(0, 0) - 2 * mat(1, 0) - mat(2, 0) + mat(0, 2) + 2 * mat(1, 2) + mat(2, 2);
				vert(i, j) = -mat(0, 0) - 2 * mat(0, 1) - mat(0, 2) + mat(2, 0) + 2 * mat(2, 1) + mat(2, 2);
			}
		}
	}
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

void GetDescriptor(const Image &hor, const Image &vert, std::vector<float> &result, bool useSse) {
	for (uint i = 0 ; i < CELL_COUNT ; ++i) {
		for (uint j = 0 ; j < CELL_COUNT ; ++j) {
			uint rows = (i == CELL_COUNT - 1) ? hor.n_rows - i * hor.n_rows / CELL_COUNT : hor.n_rows / CELL_COUNT;
			uint cols = (j == CELL_COUNT - 1) ? hor.n_cols - j * hor.n_cols / CELL_COUNT : hor.n_cols / CELL_COUNT;
			uint x = i * hor.n_rows / CELL_COUNT;
			uint y = j * hor.n_cols / CELL_COUNT;
			Image subHor = hor.submatrix(x, y, rows, cols);
			Image subVert = vert.submatrix(x, y, rows, cols);
			std::vector<float> tmp;
			if (useSse) {
				tmp = GetHist(subHor, subVert);
			}
			else {
				tmp = GetHist(subHor, subVert);
			}
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