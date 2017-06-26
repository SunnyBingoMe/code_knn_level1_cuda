#ifndef _SUNNY_F_
#define _SUNNY_F_

#pragma warning(disable : 4996)

#include <float.h>
#define Float_min FLT_MIN
#define Float_max FLT_MAX
#define Double_min DBL_MIN 
#define Double_max DBL_MAX 
#define LongDouble_min LDBL_MIN
#define LongDouble_max LDBL_MAX

/* ====================================== time related ====================================== */
#include <stdio.h>
#include <ctime>
#include <string>
void printDateNTime(){
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[20];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, 20, "%Y_%m_%d-%H:%M:%S", timeinfo);
    std::string str(buffer);

    std::cout << str << "\n";
}
/* ================================== end time related ====================================== */

/* ===================================== array related ====================================== */
void memset_float_sunny(float *p_array, float value, int itemNr){
    std::fill(p_array, p_array + itemNr, value);
}
#define GetArrayLength(p_array)  (sizeof(p_array) / sizeof((p_array)[0]))  // getArraySize works only in host code. result is always 1 in kernel code 

#define Max2_sunny(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define Min2_sunny(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

float GetArrayMax(float * array, int length) {
    float max=array[0];
    for (int i = 1 ; i < length; i++) {
        if (max<array[i]) max=array[i];
    }
    return max;
}

float GetArrayMin(float * array, int length) {
    float min=array[0];
    for (int i=1; i<length; i++) {
        if (min > array[i]) min=array[i];
    }
    return min;
}

int GetArrayMax(int * array, int length) {
    int max=array[0];
    for (int i = 1; i < length; i++) {
        if (max<array[i]) max=array[i];
    }
    return max;
}

int GetArrayMin(int * array, int length) {
    int min=array[0];
    for (int i=1; i<length; i++) {
        if (min > array[i]) min=array[i];
    }
    return min;
}
/* ================================= end array related ====================================== */

/* ====================================== file related ====================================== */

#include <fstream>

int getFileByteSizeByLocation(const char* fileLocation)
{
    std::ifstream in(fileLocation, std::ios::binary | std::ios::ate);
    return (int)in.tellg();
    // OBS: there is a bug in VS2013 for files > 4G: http://stackoverflow.com/questions/5840148/how-can-i-get-a-files-size-in-c 
}
/* ================================== end file related ====================================== */

/* ====================================== math related ====================================== */

int getOnePlaceDigit(int originalValue)
{
    return (abs(originalValue) % 10);
}

int getTenPlaceDigit(int originalValue)
{
    return ((abs(originalValue) / 10) % 10);
}

#include <math.h>
int getDigit(int originalValue, int digitIndexFromOnePlaceStart0)
{
    return (originalValue / ((int)pow(10, digitIndexFromOnePlaceStart0)) % 10);
    // ref http://bit.ly/1UprcRI 
}

/* ================================== end math related ====================================== */

#endif
