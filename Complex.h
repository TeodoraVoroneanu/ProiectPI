#pragma once

#define PI 3.14159265359
#include <cmath>

class Complex2
{
public:
	float re, im;

	Complex2(float _re = 0, float _im = 0) :re(_re), im(_im)
	{
	}

	float abs()
	{
		return sqrt(re * re + im * im);
	}

	float Re()
	{
		return re;
	}
	float Im()
	{
		return im;
	}

	Complex2 operator+(const Complex2& c)
	{
		return Complex2(re + c.re, im + c.im);
	}

	Complex2 operator*(const Complex2& c)
	{
		return Complex2(re * c.re + im * c.im, re * c.im + im * c.re);
	}

	Complex2 operator*(float a)
	{
		return Complex2(re * a, im * a);
	}

	Complex2 operator/(float a)
	{
		return Complex2(re / a, im / a);
	}

	Complex2& operator=(const Complex2& c)
	{
		re = c.re;
		im = c.im;
		return *this;
	}

	Complex2 cexp()
	{
		float e = exp(re);
		return Complex2(e * cos(im), e * sin(im));
	}
};
