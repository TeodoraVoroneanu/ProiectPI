#pragma once
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include "Complex.h"

using namespace std;
using namespace cv;


Mat Complex_to_img(Complex2* complex, int rows, int cols)
{

    unsigned char* rez = new unsigned char[rows * cols];
    float* normalizat = new float[rows * cols];

    Mat img(rows, cols, CV_8UC1, rez);


    // logaritmam valorile transformatei, pentru ca sunt prea mari (de ordinul 10^5, 10^6)
    for (int i = 0; i < rows * cols; ++i)
    {
        normalizat[i] = 1 + log(complex[i].abs());
    }

    // gasim valoarea maxima a celor logaritmate
    float max = normalizat[0];
    for (int i = 1; i < rows * cols; ++i)
    {
        if (normalizat[i] > max)
        {
            max = normalizat[i];
        }
    }

    // normalizam valorile logaritmate
    for (int i = 0; i < rows * cols; ++i)
    {
        normalizat[i] = normalizat[i] / max;
    }

    // completam intensitatile finale ale pixelilor rezultati (scalam valorile)
    for (int i = 0; i < rows * cols; ++i)
    {
        rez[i] = (unsigned char)(normalizat[i] * 255);
    }

    delete[] normalizat;
    return img;
}

//transformam o imagine din spatial in frecventa
Complex2* DFT(Mat img)
{
    cout << "------------- aplicare DFT ------------" << endl;
    int cols = img.cols;
    int rows = img.rows;
    Complex2* r = new Complex2[rows*cols];
    Complex2 element;

    Mat aux;

    if (img.type() != 5)
        img.convertTo(aux, CV_32F);
    else
        aux = img.clone();

    // pentru fiecare element din matricea rezultata
    for (int v = 0; v < rows; ++v)
    {
        for (int u = 0; u < cols; ++u)
        {
            // se calculeaza cele 2 sume
            element = Complex2(0, 0);
            for (int y = 0; y < rows; ++y)
            {
                for (int x = 0; x < cols; ++x)
                {
                    // se aduna exponentiala argumentului complex
                    element = element +
                        Complex2(0, -2 * PI * ((double)u * x / rows + (double)v * y / cols)).cexp()*
                        aux.at<float>(y,x);
                }
            }
            r[v * cols + u] = element;
            cout << "x=" << v << " y=" << u << endl;
        }
    }
    return r;
}
//


//transforma o imagine din domeniul freventei in domeniul spatial
Mat IDFT(Complex2* img,int rows,int cols)
{
    cout << "--------------------- aplicare IDFT---------------------------" << endl;
    unsigned char* rez = new unsigned char[rows*cols];
    //initializez o matrice rezultat in care locatia din care se va scrie este exact rez
    Mat rezultat(rows , cols, CV_8UC1, rez);
    Complex2* inv = new Complex2[rows * cols];
    Complex2 element;

    double* normalizat = new double[rows * cols];


    // pentru fiecare element din vectorul de frecvente
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            // se calculeaza cele 2 sume
            element = Complex2(0, 0);
            for (int v = 0; v < rows; ++v)
            {
                for (int u = 0; u < cols; ++u)
                {
                    // se aduna exponentiala argumentului complex
                    element = element + img[v * cols + u] *
                        Complex2(0, 2 * PI * ((double)u * x / rows + (double)v * y / cols)).cexp();
                        
                }
            }
            inv[(rows-y)*cols+(cols-x)] = element *(1.0/ (rows * cols));
            cout << "x=" << y << " y=" << x << endl;
        }
    }
    cout << "-------- ajuns 1";

    // normalizam valorile obtinute (intereseaza doar partea reala)
    // calculam maximul
    double max = inv[0].Re();
    for (int i = 1; i < rows * cols; ++i)
        if (inv[i].Re() > max)
            max = inv[i].Re();
    cout << "-------- ajuns 2-------------";

    // impartim fiecare valoare la maxim pentru a obtine valori in intervalul [0, 1]
    for (int i = 0; i < rows * cols; ++i) {
        normalizat[i] = inv[i].Re() / max;
        cout << normalizat[i] << " ";
    }
    cout << "-------- ajuns 3-----------";

     //inmultim fiecare valoare cu 255 pentru a scala intervalul la [0, 255]
    for (int i = 0; i < rows * cols; ++i)
        normalizat[i] *= 255;
    cout << "-------- ajuns 4";
    // punem rezultatele in vectorul de pixeli
    for (int i = 0; i < rows * cols; ++i)
        rez[i] = (unsigned char)normalizat[i];
    cout << "-------- ajuns 5";
    
    return rezultat;
}


Mat inmultire(Mat img1, Mat img2)
{

    int w1 = img1.cols;
    int h1 = img1.rows;
    cout << "w1=" << w1 << " h1=" << h1 << endl;


    int w2 = img2.cols;
    int h2 = img2.rows;
    cout << "w2=" << w2 << " h2=" << h2 << endl;

    Mat rez(h1, w2, CV_8UC1);

    for (int i = 0; i < h1; i++)
        for (int j = 0; j < w2; j++)
        {
            double element = 0;
            int y = 0;
            for (int x = 0; x < h2; x++)
            {
                element += (double)img1.at<uchar>(i, x) * (double)img2.at<uchar>(x, j);
                // cout << i << " " << j << " " << y << " " << x << " " << element << endl;
            }
            rez.at<uchar>(i, j) = element;
        }
    return rez;
}

//transforma (vector)din complex in real si returneaza rezultat
Mat getREMat(Complex2* vector, int cols, int rows)
{
    cout << "------------------------- getREMat ---------------" << endl;
    cout << "GetREmat" << endl;
    Mat rez(rows, cols, CV_32F);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
        {
            rez.at<float>(r, c) = vector[r * cols + c].Re();
        }
    return rez;
}

unsigned char* getUChars(Mat img)
{
    cout << "------------------------- getChars ---------------" << endl;

    int rows = img.rows;
    int cols = img.cols;

    unsigned char* rez = new unsigned char[rows * cols];
   

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++)
        {
            rez[r * cols + c] = img.at<uchar>(r, c);
            cout << (double) (rez[r * cols + c]) << " ";
        }
        cout << endl;
    }
    cout << endl << endl << endl;

    return rez;
}

Complex2* getComplex(Mat img) {

    cout << "------------------------- getComplex ---------------" << endl;
    cout << "GetComplex" << endl;
    int cols = img.cols;
    int rows = img.rows;
    
    Complex2* rez = new Complex2[cols * rows];
    for (int x = 0; x < rows; x++) {
        for (int y = 0; y < cols; y++) {
            rez[x * cols + y] = Complex2(img.at<float>(x, y),0);
            cout << x << " " << y << endl;
        }
    }
    return rez;
}

//Complex2* mulComplex(Complex2* img1, int rows1, int cols1, Complex2* img2, int rows2,int cols2) {
//
//    cout << "---------------Inmultire 2 X Complex2----------------" << endl;
//    cout << "w1=" << cols1 << " h1=" << rows1 << endl;
//    cout << "w2=" << cols2 << " h2=" << rows2 << endl;
//
//    Complex2* rez = new Complex2[rows1 * cols2];
//
//    for (int i = 0; i < rows1; i++)
//        for (int j = 0; j < cols1; j++)
//        {
//            Complex2 element(0,0);
//            int y = 0;
//            for (int x = 0; x < rows2; x++)
//            {
//                element = element +(img1[i*cols1+x] * img2[x*cols2+j]);
//                // cout << i << " " << j << " " << y << " " << x << " " << element << endl;
//            }
//           rez[i*cols2+j] = element;
//        }
//    return rez;
//}

Complex2* mulComplex(Complex2* img1, Complex2* img2, int rows, int cols) {

    cout << "---------------Inmultire 2 X Complex2----------------" << endl;
  /*  cout << "w1=" << cols1 << " h1=" << rows1 << endl;
    cout << "w2=" << cols2 << " h2=" << rows2 << endl;*/

    Complex2* rez = new Complex2[rows * cols];

    for (int i = 0; i < rows*cols; i++)
        rez[i] = (img1[i]*img2[i]);
    return rez;
}


void shiftare(const Mat& input, Mat& output)
{
    cout << "------------------------- shiftare ---------------" << endl;
    output = input.clone();
    int cx = output.cols / 2;
    int cy = output.rows / 2;
    Mat q0(output, Rect(0, 0, cx, cy));
    Mat q1(output, Rect(cx, 0, cx, cy));
    Mat q2(output, Rect(0, cy, cx, cy));
    Mat q3(output, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}


void calcWnrFilter(const Mat& kernel, Mat& Wiener_kernel, double SNR)
{
    cout << "------------------------- calculare filtru wiener ---------------" << endl;

    //                      H
    //  Kernel_wiener= -------------
    //                  |H|^2 + 1/SNR 
    //
    int cols = kernel.cols;
    int rows = kernel.rows;

    Mat kernel_shifted;
    shiftare(kernel, kernel_shifted);

    Complex2* K_Complex;
    K_Complex = DFT(kernel_shifted.clone());//creaza un kernel_shifted la fel,dar il pune in alta locatie

    Mat kernel_real = getREMat(K_Complex, cols, rows);
    Mat denom;
    pow(abs(kernel_real.clone()), 2, denom);
    denom += (1. / SNR);
    divide(kernel_real, denom, Wiener_kernel);
}


void deblur2Dfreq(const Mat inputImg, Mat& outputImg, const Mat Wiener_kernel)
{
    cout << "------------------------- debloring la imagine ---------------" << endl;
    Complex2* imgFreq = DFT(inputImg);
    Complex2* kernel = getComplex(Wiener_kernel);

    Complex2* deblured = mulComplex(imgFreq, kernel, Wiener_kernel.rows, Wiener_kernel.cols);

    outputImg = IDFT(deblured, inputImg.rows, Wiener_kernel.cols);
    cout << "iesit din idft";
}


//-----------imi recreeaza kernel ul ----------------------------
void calcPSF(Mat& outputImg, Size filterSize, int R)
{
    Mat h(filterSize, CV_32F, Scalar(0));
    Point point(filterSize.width / 2, filterSize.height / 2);
    circle(h, point, R, 255, -1, 8);
    Scalar summa = sum(h);
    outputImg = h / summa[0];
}




double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if (sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse = sse / (double)(I1.channels() * I1.total());
        double psnr = 10.0 * log10((255 * 255) / mse);
        return psnr;
    }
}

///=================================== folosind mai mult opencv pt Fourier si fourier invers ===============================

void deblurareWiener(const Mat& img, Mat& imgDeblurat, const Mat& H_wiener)
{
    // imgDeblurat=img*H_wiener (in freq)
   
    Mat planesImagine[2] = { Mat_<float>(img.clone()), Mat::zeros(img.size(), CV_32F) };//vine 2 matrici de memorie in care in prima imi pune partea reala,iar in a doua cea imaginara
    Mat complexImg;
    merge(planesImagine, 2, complexImg);//merge pt imagine 
    dft(complexImg, complexImg, DFT_SCALE);//aplicare fourier; rezultat in complexImg

    Mat planesH_wiener[2] = { Mat_<float>(H_wiener.clone()), Mat::zeros(H_wiener.size(), CV_32F) };
    Mat complexH_wiener;
    merge(planesH_wiener, 2, complexH_wiener);//merge pt Filtru wiener


    Mat complexImgDeblurat;
    mulSpectrums(complexImg, complexH_wiener, complexImgDeblurat, 0);// inmultire complexImg si complexH_wiener

    idft(complexImgDeblurat, complexImgDeblurat);// transformare in dom spatial
    split(complexImgDeblurat, planesImagine);//separare complex in 2 matrici
    imgDeblurat = planesImagine[0];// rezultatul e matricea cu reali
}



void calculare_filtru_Wiener(const Mat& kernel_blur, Mat& Kernel_wierner, double SNR)
{
   
    //                      H
    //  Kernel_wiener= -------------
    //                  |H|^2 + 1/SNR 
    //

    
    Mat kn_shifted;//kernelul apropimat folosit pt blurare
    shiftare(kernel_blur, kn_shifted);// inlocuire colturi kernel de blurare

    Mat plan2D[2] = { Mat_<float>(kn_shifted.clone()), Mat::zeros(kn_shifted.size(), CV_32F) };// un vector de 2 matrice real si imaginar
    Mat Kernel_blur_complex;
    merge(plan2D, 2, Kernel_blur_complex);//imbinare intr.o matrice 
    dft(Kernel_blur_complex, Kernel_blur_complex);// transformare in spatiu frecventa 

    split(Kernel_blur_complex, plan2D);// punem inapoi pe 2 matrice , de real(plan2D[0]) si imaginar(plan2D[1])
    Mat H=plan2D[0];//<------- H

    Mat numitor;
    pow(abs(H), 2, numitor);/// |H|^2
    numitor += 1./SNR;
    divide(H, numitor, Kernel_wierner);
}





