/*2016-03-24 13:39*/
%module cutils

%{
#include <iostream>
#include <limits>
#include <Python.h>
#include <cmath>
#define SWIG_FILE_WITH_INIT
using namespace std;

bool dominating(int len1, double *p1, int len2, double *p2){
    double maxDiff = -std::numeric_limits<double>::infinity();
    double minDiff = std::numeric_limits<double>::infinity();
    double maxAbsDiff = -std::numeric_limits<double>::infinity();
    int size = sizeof(p1) / sizeof(p1[0]) + 1;
    for (int i = 0; i<size; i++){
        double diff = p1[i] - p2[i];
        if (diff > maxDiff) maxDiff = diff;
        if (diff < minDiff) minDiff = diff;
        if (std::abs(diff) > maxAbsDiff) maxAbsDiff = std::abs(diff);
    }
    bool flag = maxDiff <= 0.0 && minDiff < 0.0 && !(maxAbsDiff <= 0.01);
    return flag;
}
    
    
bool stopSearch(int i, int len, double p1, double p2){
	return !(i<len && p1>=p2);	
}

double** convert(double *in, int n, int m)
{
    double **data;

    data = (double**)malloc(sizeof(*data) * n);
    for(int i = 0; i < n; ++i) {
            data[i] = in + i * m;
    }       

    return data;
}

int* calDominationCount(int poolSize, int len1, double *pool, int visitedPointsSize, int len2, double *visitedPoints, int len3, int *dominationCount){			
	double **_pool = convert((double*)pool,poolSize,len1);
	double **_visitedPoints = convert((double*)visitedPoints,visitedPointsSize,len2);	
	for(int i=0;i<poolSize;i++){
		double *p1 = _pool[i];
		int _dominationCount = 0;
		int j = 0;
		bool flag = true;
		while (flag){
			double *p2 = _visitedPoints[j];
			if (dominating(len2,p2,len1,p1)) {_dominationCount += 1;}
			j += 1;
			if (stopSearch(j,visitedPointsSize,p1[0],p2[0])) {flag = false;}			
		}
		dominationCount[i] = _dominationCount; 
	}
	return dominationCount;
}

%}

%include "numpy.i"

%init %{
	import_array();
%}

/*  typemaps for the two arrays, the second will be modified in-place */
%apply (int DIM1, double* IN_ARRAY1) {(int len1, double* p1), (int len2, double* p2)};
%apply (int DIM1, int DIM2, double* IN_ARRAY2) {(int poolSize, int len1, double *pool),(int visitedPointsSize, int len2, double *visitedPoints)};
%apply (int DIM1, int* ARGOUT_ARRAY1) {(int len3, int * dominationCount)};

bool dominating(int len1, double *p1, int len2, double *p2);
bool stopSearch(int i, int len, double p1, double p2);
double** convert(double *in, int n, int m);
int* calDominationCount(int poolSize, int len1, double *pool, int visitedPointsSize, int len2, double *visitedPoints, int len3, int *dominationCount);
