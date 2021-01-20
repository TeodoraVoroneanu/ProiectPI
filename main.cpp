#include <QApplication.h>
#include "ImageGrid.h"
#include "operatii.h"



int main(int argc, char* argv[])
{

	QApplication a(argc, argv);

	ImageGrid* grid = new ImageGrid("Deblurare folosind Wiener filter");

	// ----------------------------------- deschidere imagini ------------
	string image_path = samples::findFile("Images/pisica.jpg");
	Mat img = imread(image_path, IMREAD_GRAYSCALE);
	int rows = img.rows;
	int cols = img.cols;

	if (img.empty())
	{
		std::cout << "Could not read the image: " << std::endl;
		return 1;
	}
	int R = 7;
	Mat blur;// imaginea blurata 
	GaussianBlur(img, blur, Size(R, R), 3);

	double psnr = getPSNR(img, blur);// o aproximare de SNR

	Rect rec = Rect(0, 0, img.cols & -2, img.rows & -2);
	Mat kernel;
	calcPSF(kernel, rec.size(), (R / 2) + 1);//refacere kernel de blurare

	// ========================== folosing Fourier si invers Fourier fara opencv =======================================


	/*grid->addImage(getUChars(img), rows, cols, 0, 0, "Imagine original");
	grid->addImage(getUChars(blur), rows, cols, 0, 1, "Imagine blurata");


	Mat Wiener_kernel;
	calcWnrFilter(kernel, Wiener_kernel, psnr);
	
	Mat afis_Wiener;
	Wiener_kernel.convertTo(afis_Wiener, CV_8UC1);
	normalize(afis_Wiener, afis_Wiener, 0, 255, NORM_MINMAX);
	grid->addImage(getUChars(afis_Wiener), afis_Wiener.rows, afis_Wiener.cols, 1, 0, "Filtrul Wiener");
	cout << Wiener_kernel.type()<<"----------"<<endl;
	
	
	Mat rez;
	deblur2Dfreq(img, rez, Wiener_kernel);
	cout << "am ajums aici";
	grid->addImage(getUChars(rez), rez.rows, rez.cols, 1, 1, "rezultat");*/
	// ====================================== folosind dft si idft din opencv ==================================
	imshow("Imagine original", img);
	imshow("Imagine blurata", blur);

	Mat wiener;
	calculare_filtru_Wiener(kernel, wiener, psnr);
	Mat wiere_sp;
	imshow("Filtru wiener", wiener);

	Mat imgDeblurat;
	deblurareWiener(img, imgDeblurat, wiener);

	Mat rezultat;
	imgDeblurat.convertTo(rezultat, CV_8U);
	normalize(rezultat, rezultat, 0, 255, NORM_MINMAX);
	imshow("Imagine deblurata", rezultat);

	waitKey(0);

	grid->show();
	return a.exec();
}

