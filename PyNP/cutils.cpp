#include <iostream>
#include <limits>
#include <Python.h>
#define SWIG_FILE_WITH_INIT
using namespace std;

bool dominating(int len1, double *p1, int len2, double *p2){
	double maxDiff = -std::numeric_limits<double>::infinity();
	double minDiff = std::numeric_limits<double>::infinity();
	int size = sizeof(p1) / sizeof(p1[0]) + 1;
	for (int i = 0; i<size; i++){
		double diff = p1[i] - p2[i];
		if (diff > maxDiff) maxDiff = diff;
		if (diff < minDiff) minDiff = diff;
	}
	bool flag = maxDiff <= 0.0 && minDiff < 0.0;
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


/*
int main(){
    
    int poolSize = 2;
    int visitedPointsSize = 4;
	
    double **pool;
    pool = new double*[2];
    for(int i=0;i<2;i++){
        pool[i] = new double[2];
    }
    pool[0][0] = 1.1;
    pool[0][1] = 2.3;
    pool[1][0] = 2.2;    
    pool[1][1] = 0.5;
	
    double **visitedPoints;
    visitedPoints = new double*[4];
    for(int i=0;i<2;i++){
        visitedPoints[i] = new double[2];
    }
    visitedPoints[0][0] = 1.1;
    visitedPoints[0][1] = 2.3;
    visitedPoints[1][0] = 2.2;    
    visitedPoints[1][1] = 0.5;	
	visitedPoints[2][0] = 1.5;
	visitedPoints[2][1] = 2.5;
	visitedPoints[3][0] = 3.5;
	visitedPoints[3][1] = 1.5;
	
    int *dominationCount;
    dominationCount = calDominationCount(poolSize,pool,visitedPointsSize,visitedPoints);
    for(int i=0;i<poolSize;i++){
        cout<<i<<" "<<dominationCount[i]<<endl;    
    }
    
    
}

*/