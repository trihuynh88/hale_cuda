#include <iostream>
#include <Hale.h>
#include <glm/glm.hpp>

#include "unistd.h" // for sleep()

#include <fstream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "lib/Image.h"
#include <vector>
#include <unordered_map>

using namespace std;

//from cuda_volume_rendering
#define PI 3.14159265

#define MAX(a,b) ((a)>(b)?(a):(b))


texture<float, 3, cudaReadModeElementType> tex0;  // 3D texture
texture<float, 3, cudaReadModeElementType> tex1;  // 3D texture
texture<float, 3, cudaReadModeElementType> tex2;  // 3D texture
/*
cudaArray *d_volumeArray0 = 0;
cudaArray *d_volumeArray1 = 0;
cudaArray *d_volumeArray2 = 0;
*/
#define NTEX 7
//texture<float, 3, cudaReadModeElementType> tex[NTEX];
//+1 for an extra volume for interpolation
cudaArray *d_volumeArray[NTEX+1];

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__
float w0(float a)
{
    return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);
}

__host__ __device__
float w1(float a)
{
    return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__
float w2(float a)
{
    return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__
float w3(float a)
{
    return (1.0f/6.0f)*(a*a*a);
}

//derivatives of basic functions
__host__ __device__
float w0g(float a)
{
    return -(1.0f/2.0f)*a*a + a - (1.0f/2.0f);
}

__host__ __device__
float w1g(float a)
{

    return (3.0f/2.0f)*a*a - 2*a;
}

__host__ __device__
float w2g(float a)
{
    return -(3.0f/2.0f)*a*a + a + (1.0/2.0);
}

__host__ __device__
float w3g(float a)
{
    return (1.0f/2.0f)*a*a;
}

//second derivatives of basic functions
__host__ __device__
float w0gg(float a)
{
    return 1-a;
}

__host__ __device__
float w1gg(float a)
{

    return 3*a-2;
}

__host__ __device__
float w2gg(float a)
{
    return 1-3*a;
}

__host__ __device__
float w3gg(float a)
{
    return a;
}



// filter 4 values using cubic splines
template<class T>
__host__ __device__
T cubicFilter(float x, T c0, T c1, T c2, T c3)
{
    T r;
    r = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}

//filtering with derivative of basic functions
template<class T>
__host__ __device__
T cubicFilter_G(float x, T c0, T c1, T c2, T c3)
{
    T r;
    r = c0 * w0g(x);
    r += c1 * w1g(x);
    r += c2 * w2g(x);
    r += c3 * w3g(x);
    return r;
}

//filtering with second derivative of basic functions
template<class T>
__host__ __device__
T cubicFilter_GG(float x, T c0, T c1, T c2, T c3)
{
    T r;
    r = c0 * w0gg(x);
    r += c1 * w1gg(x);
    r += c2 * w2gg(x);
    r += c3 * w3gg(x);
    return r;
}


template<class T, class R>  // texture data type, return type
__device__
R tex3DBicubicXY(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return cubicFilter<R>(fy,
                          cubicFilter<R>(fx, tex3D(texref, px-1, py-1,z), tex3D(texref, px, py-1,z), tex3D(texref, px+1, py-1,z), tex3D(texref, px+2,py-1,z)),
                          cubicFilter<R>(fx, tex3D(texref, px-1, py,z),   tex3D(texref, px, py,z),   tex3D(texref, px+1, py,z),   tex3D(texref, px+2, py,z)),
                          cubicFilter<R>(fx, tex3D(texref, px-1, py+1,z), tex3D(texref, px, py+1,z), tex3D(texref, px+1, py+1,z), tex3D(texref, px+2, py+1,z)),
                          cubicFilter<R>(fx, tex3D(texref, px-1, py+2,z), tex3D(texref, px, py+2,z), tex3D(texref, px+1, py+2,z), tex3D(texref, px+2, py+2,z))
                         );
}

//gradient in X direction
template<class T, class R>  // texture data type, return type
__device__
R tex3DBicubicXY_GX(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return cubicFilter<R>(fy,
                          cubicFilter_G<R>(fx, tex3D(texref, px-1, py-1,z), tex3D(texref, px, py-1,z), tex3D(texref, px+1, py-1,z), tex3D(texref, px+2,py-1,z)),
                          cubicFilter_G<R>(fx, tex3D(texref, px-1, py,z),   tex3D(texref, px, py,z),   tex3D(texref, px+1, py,z),   tex3D(texref, px+2, py,z)),
                          cubicFilter_G<R>(fx, tex3D(texref, px-1, py+1,z), tex3D(texref, px, py+1,z), tex3D(texref, px+1, py+1,z), tex3D(texref, px+2, py+1,z)),
                          cubicFilter_G<R>(fx, tex3D(texref, px-1, py+2,z), tex3D(texref, px, py+2,z), tex3D(texref, px+1, py+2,z), tex3D(texref, px+2, py+2,z))
                         );
}

template<class T, class R>  // texture data type, return type
__device__
R tex3DBicubicXY_GY(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return cubicFilter_G<R>(fy,
                          cubicFilter<R>(fx, tex3D(texref, px-1, py-1,z), tex3D(texref, px, py-1,z), tex3D(texref, px+1, py-1,z), tex3D(texref, px+2,py-1,z)),
                          cubicFilter<R>(fx, tex3D(texref, px-1, py,z),   tex3D(texref, px, py,z),   tex3D(texref, px+1, py,z),   tex3D(texref, px+2, py,z)),
                          cubicFilter<R>(fx, tex3D(texref, px-1, py+1,z), tex3D(texref, px, py+1,z), tex3D(texref, px+1, py+1,z), tex3D(texref, px+2, py+1,z)),
                          cubicFilter<R>(fx, tex3D(texref, px-1, py+2,z), tex3D(texref, px, py+2,z), tex3D(texref, px+1, py+2,z), tex3D(texref, px+2, py+2,z))
                         );
}

template<class T, class R>
__device__
R tex3DBicubic(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float pz = floor(z);
    float fz = z - pz;
    return cubicFilter<R>(fz,
                          tex3DBicubicXY<T,R>(texref,x,y,pz-1),
                          tex3DBicubicXY<T,R>(texref,x,y,pz),
                          tex3DBicubicXY<T,R>(texref,x,y,pz+1),
                          tex3DBicubicXY<T,R>(texref,x,y,pz+2)
                          );
}

template<class T, class R>
__device__
R tex3DBicubic_GX(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float pz = floor(z);
    float fz = z - pz;
    return cubicFilter<R>(fz,
                          tex3DBicubicXY_GX<T,R>(texref,x,y,pz-1),
                          tex3DBicubicXY_GX<T,R>(texref,x,y,pz),
                          tex3DBicubicXY_GX<T,R>(texref,x,y,pz+1),
                          tex3DBicubicXY_GX<T,R>(texref,x,y,pz+2)
                          );
}

template<class T, class R>
__device__
R tex3DBicubic_GY(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float pz = floor(z);
    float fz = z - pz;
    return cubicFilter<R>(fz,
                          tex3DBicubicXY_GY<T,R>(texref,x,y,pz-1),
                          tex3DBicubicXY_GY<T,R>(texref,x,y,pz),
                          tex3DBicubicXY_GY<T,R>(texref,x,y,pz+1),
                          tex3DBicubicXY_GY<T,R>(texref,x,y,pz+2)
                          );
}

template<class T, class R>
__device__
R tex3DBicubic_GZ(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float pz = floor(z);
    float fz = z - pz;
    return cubicFilter_G<R>(fz,
                            tex3DBicubicXY<T,R>(texref,x,y,pz-1),
                            tex3DBicubicXY<T,R>(texref,x,y,pz),
                            tex3DBicubicXY<T,R>(texref,x,y,pz+1),
                            tex3DBicubicXY<T,R>(texref,x,y,pz+2)
                            );
}

template<class T, class R>  // texture data type, return type
__device__
R tex3DBicubicXY_GGX(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return cubicFilter<R>(fy,
                          cubicFilter_GG<R>(fx, tex3D(texref, px-1, py-1,z), tex3D(texref, px, py-1,z), tex3D(texref, px+1, py-1,z), tex3D(texref, px+2,py-1,z)),
                          cubicFilter_GG<R>(fx, tex3D(texref, px-1, py,z),   tex3D(texref, px, py,z),   tex3D(texref, px+1, py,z),   tex3D(texref, px+2, py,z)),
                          cubicFilter_GG<R>(fx, tex3D(texref, px-1, py+1,z), tex3D(texref, px, py+1,z), tex3D(texref, px+1, py+1,z), tex3D(texref, px+2, py+1,z)),
                          cubicFilter_GG<R>(fx, tex3D(texref, px-1, py+2,z), tex3D(texref, px, py+2,z), tex3D(texref, px+1, py+2,z), tex3D(texref, px+2, py+2,z))
                         );
}

template<class T, class R>  // texture data type, return type
__device__
R tex3DBicubicXY_GGY(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return cubicFilter_GG<R>(fy,
                          cubicFilter<R>(fx, tex3D(texref, px-1, py-1,z), tex3D(texref, px, py-1,z), tex3D(texref, px+1, py-1,z), tex3D(texref, px+2,py-1,z)),
                          cubicFilter<R>(fx, tex3D(texref, px-1, py,z),   tex3D(texref, px, py,z),   tex3D(texref, px+1, py,z),   tex3D(texref, px+2, py,z)),
                          cubicFilter<R>(fx, tex3D(texref, px-1, py+1,z), tex3D(texref, px, py+1,z), tex3D(texref, px+1, py+1,z), tex3D(texref, px+2, py+1,z)),
                          cubicFilter<R>(fx, tex3D(texref, px-1, py+2,z), tex3D(texref, px, py+2,z), tex3D(texref, px+1, py+2,z), tex3D(texref, px+2, py+2,z))
                         );
}

//derivative through X, then through Y
template<class T, class R>  // texture data type, return type
__device__
R tex3DBicubicXY_GYGX(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    return cubicFilter_G<R>(fy,
                          cubicFilter_G<R>(fx, tex3D(texref, px-1, py-1,z), tex3D(texref, px, py-1,z), tex3D(texref, px+1, py-1,z), tex3D(texref, px+2,py-1,z)),
                          cubicFilter_G<R>(fx, tex3D(texref, px-1, py,z),   tex3D(texref, px, py,z),   tex3D(texref, px+1, py,z),   tex3D(texref, px+2, py,z)),
                          cubicFilter_G<R>(fx, tex3D(texref, px-1, py+1,z), tex3D(texref, px, py+1,z), tex3D(texref, px+1, py+1,z), tex3D(texref, px+2, py+1,z)),
                          cubicFilter_G<R>(fx, tex3D(texref, px-1, py+2,z), tex3D(texref, px, py+2,z), tex3D(texref, px+1, py+2,z), tex3D(texref, px+2, py+2,z))
                         );
}

template<class T, class R>
__device__
R tex3DBicubic_GGX(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float pz = floor(z);
    float fz = z - pz;
    return cubicFilter<R>(fz,
                          tex3DBicubicXY_GGX<T,R>(texref,x,y,pz-1),
                          tex3DBicubicXY_GGX<T,R>(texref,x,y,pz),
                          tex3DBicubicXY_GGX<T,R>(texref,x,y,pz+1),
                          tex3DBicubicXY_GGX<T,R>(texref,x,y,pz+2)
                          );
}

template<class T, class R>
__device__
R tex3DBicubic_GGY(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float pz = floor(z);
    float fz = z - pz;
    return cubicFilter<R>(fz,
                          tex3DBicubicXY_GGY<T,R>(texref,x,y,pz-1),
                          tex3DBicubicXY_GGY<T,R>(texref,x,y,pz),
                          tex3DBicubicXY_GGY<T,R>(texref,x,y,pz+1),
                          tex3DBicubicXY_GGY<T,R>(texref,x,y,pz+2)
                          );
}

template<class T, class R>
__device__
R tex3DBicubic_GGZ(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float pz = floor(z);
    float fz = z - pz;
    return cubicFilter_GG<R>(fz,
                            tex3DBicubicXY<T,R>(texref,x,y,pz-1),
                            tex3DBicubicXY<T,R>(texref,x,y,pz),
                            tex3DBicubicXY<T,R>(texref,x,y,pz+1),
                            tex3DBicubicXY<T,R>(texref,x,y,pz+2)
                            );
}

//derivative through X, then through Y
template<class T, class R>
__device__
R tex3DBicubic_GYGX(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float pz = floor(z);
    float fz = z - pz;
    return cubicFilter<R>(fz,
                          tex3DBicubicXY_GYGX<T,R>(texref,x,y,pz-1),
                          tex3DBicubicXY_GYGX<T,R>(texref,x,y,pz),
                          tex3DBicubicXY_GYGX<T,R>(texref,x,y,pz+1),
                          tex3DBicubicXY_GYGX<T,R>(texref,x,y,pz+2)
                          );
}

//derivative through X, then through Z
template<class T, class R>
__device__
R tex3DBicubic_GZGX(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float pz = floor(z);
    float fz = z - pz;
    return cubicFilter_G<R>(fz,
                          tex3DBicubicXY_GX<T,R>(texref,x,y,pz-1),
                          tex3DBicubicXY_GX<T,R>(texref,x,y,pz),
                          tex3DBicubicXY_GX<T,R>(texref,x,y,pz+1),
                          tex3DBicubicXY_GX<T,R>(texref,x,y,pz+2)
                          );
}

//derivative through Y, then through Z
template<class T, class R>
__device__
R tex3DBicubic_GZGY(const texture<T, 3, cudaReadModeElementType> texref, float x, float y, float z)
{
    float pz = floor(z);
    float fz = z - pz;
    return cubicFilter_G<R>(fz,
                          tex3DBicubicXY_GY<T,R>(texref,x,y,pz-1),
                          tex3DBicubicXY_GY<T,R>(texref,x,y,pz),
                          tex3DBicubicXY_GY<T,R>(texref,x,y,pz+1),
                          tex3DBicubicXY_GY<T,R>(texref,x,y,pz+2)
                          );
}


__host__ __device__
int cu_getIndex2(int i, int j, int s1, int s2)
{
    return i*s2+j;
}

__host__ __device__
double dotProduct(double *u, double *v, int s)
{
    double result = 0;
    for (int i=0; i<s; i++)
        result += (u[i]*v[i]);
    return result;
}

__host__ __device__
double lenVec(double *a, int s)
{
    double len = 0;
    for (int i=0; i<s; i++)
        len += (a[i]*a[i]);
    len = sqrt(len);
    return len;
}

__host__ __device__
void addVector(double *a, double *b, double *c, int len)
{
  for (int i=0; i<len; i++)
    c[i] = a[i]+b[i];
}

__host__ __device__
void scaleVector(double *a, int len, double scale)
{
  for (int i=0; i<len; i++)
    a[i]*=scale;
}

void mulMatPoint(double X[4][4], double Y[4], double Z[4])
{
    for (int i=0; i<4; i++)
        Z[i] = 0;

    for (int i=0; i<4; i++)
        for (int k=0; k<4; k++)
            Z[i] += (X[i][k]*Y[k]);
}


__device__
void cu_mulMatPoint(double* X, double* Y, double* Z)
{
    for (int i=0; i<4; i++)
        Z[i] = 0;

    for (int i=0; i<4; i++)
        for (int k=0; k<4; k++)
            Z[i] += (X[cu_getIndex2(i,k,4,4)]*Y[k]);
}

__device__
void cu_mulMatPoint3(double* X, double* Y, double* Z)
{
    for (int i=0; i<3; i++)
        Z[i] = 0;

    for (int i=0; i<3; i++)
        for (int k=0; k<3; k++)
            Z[i] += (X[cu_getIndex2(i,k,3,3)]*Y[k]);
}

__host__ __device__
void advancePoint(double* point, double* dir, double scale, double* newpos)
{
    for (int i=0; i<3; i++)
        newpos[i] = point[i]+dir[i]*scale;
}

__device__
bool cu_isInsideDouble(double i, double j, double k, int dim1, int dim2, int dim3)
{
    return ((i>=0)&&(i<=(dim1-1))&&(j>=0)&&(j<=(dim2-1))&&(k>=0)&&(k<=(dim3-1)));
}

__device__
double cu_computeAlpha(double val, double grad_len, double isoval, double alphamax, double thickness)
{
    if ((grad_len == 0.0) && (val == isoval))
        return alphamax;
    else
        if ((grad_len>0.0) && (isoval >= (val-thickness*grad_len)) && (isoval <= (val+thickness*grad_len)))
            return alphamax*(1-abs(isoval-val)/(grad_len*thickness));
        else
            return 0.0;
}

__device__
double cu_inAlpha(double val, double grad_len, double isoval, double thickness)
{
    if (val >= isoval)
        return 1.0;
    else
    {
        return max(0.0,(1-abs(isoval-val)/(grad_len*thickness)));
    }
}

__device__
double cu_inAlphaX(double dis, double thickness)
{
    if (dis<0)
        return 1.0;
    return max(0.0,min(1.0,1.4-fabs(dis)/thickness));
}

__host__ __device__
void normalize(double *a, int s)
{
    double len = lenVec(a,s);
    for (int i=0; i<s; i++)
        a[i] = a[i]/len;
}

__host__ __device__
double diss2P(double x1,double y1,double z1,double x2,double y2,double z2)
{
    double dis1 = x2-x1;
    double dis2 = y2-y1;
    double dis3 = z2-z1;
    return (dis1*dis1+dis2*dis2+dis3*dis3);
}

__host__ __device__
void mulMat3(double* X, double* Y, double* Z)
{
    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
        {
            for (int k=0; k<3; k++)
            {
                Z[cu_getIndex2(i,j,3,3)] += (X[cu_getIndex2(i,k,3,3)]*Y[cu_getIndex2(k,j,3,3)]);
            }
        }
}

__host__ __device__
void invertMat33(double X[][3], double Y[][3])
{
    double det = X[0][0]* (X[1][1]* X[2][2]- X[2][1]* X[1][2])-
        X[0][1]* (X[1][0]* X[2][2]- X[1][2]* X[2][0])+
        X[0][2]* (X[1][0]* X[2][1]- X[1][1]* X[2][0]);

    double invdet = 1 / det;

    Y[0][0]= (X[1][1]* X[2][2]- X[2][1]* X[1][2]) * invdet;
    Y[0][1]= (X[0][2]* X[2][1]- X[0][1]* X[2][2]) * invdet;
    Y[0][2]= (X[0][1]* X[1][2]- X[0][2]* X[1][1])* invdet;
    Y[1][0]= (X[1][2]* X[2][0]- X[1][0]* X[2][2])* invdet;
    Y[1][1]= (X[0][0]* X[2][2]- X[0][2]* X[2][0])* invdet;
    Y[1][2]= (X[1][0]* X[0][2]- X[0][0]* X[1][2])* invdet;
    Y[2][0]= (X[1][0]* X[2][1]- X[2][0]* X[1][1])* invdet;
    Y[2][1]= (X[2][0]* X[0][1]- X[0][0]* X[2][1])* invdet;
    Y[2][2]= (X[0][0]* X[1][1]- X[1][0]* X[0][1]) * invdet;
}

__host__ __device__
void eigenOfHess(double* hessian, double *eigval)
{
  double Dxx = hessian[cu_getIndex2(0,0,3,3)];
  double Dyy = hessian[cu_getIndex2(1,1,3,3)];
  double Dzz = hessian[cu_getIndex2(2,2,3,3)];
  double Dxy = hessian[cu_getIndex2(0,1,3,3)];
  double Dxz = hessian[cu_getIndex2(0,2,3,3)];
  double Dyz = hessian[cu_getIndex2(1,2,3,3)];

  double J1 = Dxx + Dyy + Dzz;
  double J2 = Dxx*Dyy + Dxx*Dzz + Dyy*Dzz - Dxy*Dxy - Dxz*Dxz - Dyz*Dyz;
  double J3 = 2*Dxy*Dxz*Dyz + Dxx*Dyy*Dzz - Dxz*Dxz*Dyy - Dxx*Dyz*Dyz - Dxy*Dxy*Dzz;
  double Q = (J1*J1-3*J2)/9;
  double R = (-9*J1*J2+27*J3+2*J1*J1*J1)/54;
  double theta = (1.0/3.0)*acos(R/sqrt(Q*Q*Q));
  double sqrtQ = sqrt(Q);
  double twosqrtQ = 2*sqrtQ;
  double J1o3 = J1/3;
  eigval[0] = J1o3 + twosqrtQ*cos(theta);
  eigval[1] = J1o3 + twosqrtQ*cos(theta-2*M_PI/3);
  eigval[2] = J1o3 + twosqrtQ*cos(theta+2*M_PI/3);
}

__device__
void computeHessian(double *hessian, double *p)
{
  hessian[cu_getIndex2(0,0,3,3)]=tex3DBicubic_GGX<float,float>(tex0,p[0],p[1],p[2]);
  hessian[cu_getIndex2(0,1,3,3)]=tex3DBicubic_GYGX<float,float>(tex0,p[0],p[1],p[2]);
  hessian[cu_getIndex2(0,2,3,3)]=tex3DBicubic_GZGX<float,float>(tex0,p[0],p[1],p[2]);
  hessian[cu_getIndex2(1,1,3,3)]=tex3DBicubic_GGY<float,float>(tex0,p[0],p[1],p[2]);
  hessian[cu_getIndex2(1,2,3,3)]=tex3DBicubic_GZGY<float,float>(tex0,p[0],p[1],p[2]);
  hessian[cu_getIndex2(2,2,3,3)]=tex3DBicubic_GGZ<float,float>(tex0,p[0],p[1],p[2]);

  hessian[cu_getIndex2(1,0,3,3)] = hessian[cu_getIndex2(0,1,3,3)];
  hessian[cu_getIndex2(2,0,3,3)] = hessian[cu_getIndex2(0,2,3,3)];
  hessian[cu_getIndex2(2,1,3,3)] = hessian[cu_getIndex2(1,2,3,3)];  
}

__host__ __device__
void cross(double *u, double *v, double *w)
{
    w[0] = u[1]*v[2]-u[2]*v[1];
    w[1] = u[2]*v[0]-u[0]*v[2];
    w[2] = u[0]*v[1]-u[1]*v[0];
}

__host__ __device__
float lerp(float y0, float y1, float x0, float x, float x1)
{
  float alpha = (x-x0)/(x1-x0);
  return y0*(1-alpha)+alpha*y1;
}

__host__ __device__
float lerp(float y0, float y1, float alpha)
{
  return y0*(1-alpha)+alpha*y1;
}

//interpolate the volume in between
__global__
void kernel_interpol(float *intervol, int* dim, float alpha)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;
    int k = (blockIdx.z * blockDim.z) + threadIdx.z;

    if ((i>=dim[1]) || (j>=dim[2]) || (k>=dim[3]))
        return;

    intervol[k*dim[2]*dim[1] + j*dim[1] + i] = lerp(tex3D(tex0,i,j,k),tex3D(tex1,i,j,k),alpha);
    if (i<=2 && j<=2 && k<=2)
    {
      printf("inside kernel_interpol, val at (%d,%d,%d) = %f\n", i,j,k,intervol[k*dim[2]*dim[1] + j*dim[1] + i]);
      printf("inside kernel_interpol, tex0 at (%d,%d,%d) = %f\n",i,j,k, tex3D(tex0,i,j,k));
    }
}

//currently working in index-space
//do MIP for a small slice around each point
__global__
void kernel_cpr(int* dim, int *size, double verextent, double *center, double *dir1, double *dir2, double swidth, double sstep, int nOutChannel, double* imageDouble
        )
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ((i>=size[0]) || (j>=size[1]))
        return;

    double pixsize = verextent/size[1];
    int ni = i-size[0]/2;
    int nj = size[1]/2 - j;
    double pointi[3];
    advancePoint(center,dir1,ni*pixsize,pointi);
    advancePoint(pointi,dir2,nj*pixsize,pointi);

    double mipdir[3];
    cross(dir1,dir2,mipdir);
    normalize(mipdir,3);

    double mipval = INT_MIN;

    
    double curpoint[3];
    int k;
    for (k=0; k<3; k++)
      curpoint[k] = pointi[k] - mipdir[k]*swidth/2;

    for (k=0; k<ceil(swidth/sstep); k++)
    {
      double curval;
      //curval = tex3DBicubic<float,float>(tex0,curpoint[0],curpoint[1],curpoint[2]);    
      curval = tex3DBicubic<float,float>(tex2,curpoint[0],curpoint[1],curpoint[2]);    
      mipval = MAX(mipval,curval);
      curpoint[0] = curpoint[0] + mipdir[0]*sstep;
      curpoint[1] = curpoint[1] + mipdir[1]*sstep;
      curpoint[2] = curpoint[2] + mipdir[2]*sstep;
    }
        
    imageDouble[j*size[0]*nOutChannel+i*nOutChannel] = 0;
    imageDouble[j*size[0]*nOutChannel+i*nOutChannel+1] = mipval;
    for (int k=2; k<nOutChannel-1; k++)
      imageDouble[j*size[0]*nOutChannel+i*nOutChannel+k] = 0;
    imageDouble[j*size[0]*nOutChannel+i*nOutChannel+nOutChannel-1] = 1;   
}

__global__
void kernel_cprinter(double alpha, int* dim, int *size, double verextent, double *center, double *dir1, double *dir2, double swidth, double sstep, int nOutChannel, double* imageDouble
        )
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ((i>=size[0]) || (j>=size[1]))
        return;

    double pixsize = verextent/size[1];
    int ni = i-size[0]/2;
    int nj = size[1]/2 - j;
    double pointi[3];
    advancePoint(center,dir1,ni*pixsize,pointi);
    advancePoint(pointi,dir2,nj*pixsize,pointi);

    double mipdir[3];
    cross(dir1,dir2,mipdir);
    normalize(mipdir,3);

    double mipval = INT_MIN;

    
    double curpoint[3];
    int k;
    for (k=0; k<3; k++)
      curpoint[k] = pointi[k] - mipdir[k]*swidth/2;

    for (k=0; k<ceil(swidth/sstep); k++)
    {
      double curval;  
      curval = lerp(tex3DBicubic<float,float>(tex0,curpoint[0],curpoint[1],curpoint[2]),tex3DBicubic<float,float>(tex1,curpoint[0],curpoint[1],curpoint[2]),alpha);
      mipval = MAX(mipval,curval);
      curpoint[0] = curpoint[0] + mipdir[0]*sstep;
      curpoint[1] = curpoint[1] + mipdir[1]*sstep;
      curpoint[2] = curpoint[2] + mipdir[2]*sstep;
    }
        
    imageDouble[j*size[0]*nOutChannel+i*nOutChannel] = 0;
    imageDouble[j*size[0]*nOutChannel+i*nOutChannel+1] = mipval;
    for (int k=2; k<nOutChannel-1; k++)
      imageDouble[j*size[0]*nOutChannel+i*nOutChannel+k] = 0;
    imageDouble[j*size[0]*nOutChannel+i*nOutChannel+nOutChannel-1] = 1;   
}


void computeMean(double *points, int n, double *means)
{
  memset(means,0,sizeof(double)*3);
  for (int i=0; i<3; i++)
  {
    for (int k=0; k<n; k++)
      means[i] += points[k*3+i];
    means[i]/=n;
  }
}

void computeCovariance(double *points, int n, double *cov)
{
  double means[3];
  computeMean(points,n,means);
  //memset(cov,0,sizeof(double)*9);
  for (int i=0; i<3; i++)
    for (int j=i; j<3; j++)
    {
      double localcov = 0;
      for (int k=0; k<n; k++)
      {
        localcov += (points[k*3+i]-means[i])*(points[k*3+j]-means[j]);
      }
      localcov/=n;
      cov[cu_getIndex2(i,j,3,3)] = cov[cu_getIndex2(j,i,3,3)] = localcov;
    }
}

int isScaleOf(double *v1, double *v2, int s)
{
  double factor;
  for (int i=0; i<s; i++)
    if (v1[i])
    {
      factor = v2[i]/v1[i];
      break;
    }
  for (int i=0; i<s; i++)
    if (v1[i]*factor != v2[i])
      return 0;
  return 1;
}

//for symmetric 3x3 matrix
void computeEigenVec(double *matrix, double eigval, double *eigvec)
{
  double matrixtmp[9];
  memcpy(matrixtmp,matrix,sizeof(double)*9);
  for (int i=0; i<3; i++)
    matrixtmp[cu_getIndex2(i,i,3,3)] = matrixtmp[cu_getIndex2(i,i,3,3)] - eigval;

  double col1[3], col2[3];
  int ind = 0;
  for (ind = 0; ind<3; ind++)
  {
    if (matrixtmp[cu_getIndex2(0,ind,3,3)] || matrixtmp[cu_getIndex2(1,ind,3,3)] || matrixtmp[cu_getIndex2(2,ind,3,3)])
      break;
  }
  if (ind<3)
  {
    for (int i=0; i<3; i++)
      col1[i] = matrixtmp[cu_getIndex2(i,ind,3,3)];
    int ind2;
    for (ind2 = ind+1; ind2<3; ind2++)
    {
      if (matrixtmp[cu_getIndex2(0,ind2,3,3)] || matrixtmp[cu_getIndex2(1,ind2,3,3)] || matrixtmp[cu_getIndex2(2,ind2,3,3)])
        break;   
    }
    if (ind2<3)
    {
      for (int i=0; i<3; i++)
        col2[i] = matrixtmp[cu_getIndex2(i,ind2,3,3)];
      if (isScaleOf(col1,col2,3))
      {
        ind2++;
        if (ind2<3)
        {
          if (matrixtmp[cu_getIndex2(0,ind2,3,3)] || matrixtmp[cu_getIndex2(1,ind2,3,3)] || matrixtmp[cu_getIndex2(2,ind2,3,3)])
          {
            for (int i=0; i<3; i++)
              col2[i] = matrixtmp[cu_getIndex2(i,ind2,3,3)];
            if (isScaleOf(col1,col2,3))
            {
              double tmp[3];
              memcpy(tmp,col1,sizeof(double)*3);
              tmp[0]++;
              double tmp2[3];
              cross(col1,tmp,tmp2);
              cross(tmp2,col1,eigvec);
            }
            else
            {
              cross(col1,col2,eigvec);
            }
          }
          else
          {
            double tmp[3];
            memcpy(tmp,col1,sizeof(double)*3);
            tmp[0]++;
            double tmp2[3];
            cross(col1,tmp,tmp2);
            cross(tmp2,col1,eigvec);
          }
        }
        else
        {
          double tmp[3];
          memcpy(tmp,col1,sizeof(double)*3);
          tmp[0]++;
          double tmp2[3];
          cross(col1,tmp,tmp2);
          cross(tmp2,col1,eigvec);
        }
      }
      else
      {
        cross(col1,col2,eigvec);
      }
    }
    else
    {
      double tmp[3];
      memcpy(tmp,col1,sizeof(double)*3);
      tmp[0]++;
      double tmp2[3];
      cross(col1,tmp,tmp2);
      cross(tmp2,col1,eigvec);
    }
  }
  else
  {
    eigvec[0] = eigvec[1] = eigvec[2] = 1;
  }
  normalize(eigvec,3);
}

void drawCircle(unsigned char *img, int s0, int s1, int s2, int drawchan, int c1, int c2, double rad)
{
  double angstep = 0.2;
  for (double curang = 0; curang<2*M_PI; curang+=angstep)
  {
    int i1, i2;
    i2 = sin(curang)*rad;
    i1 = cos(curang)*rad;
    i1 += c1;
    i2 += c2;

    img[i2*s1*s0 + i1*s0 + drawchan] = 255;
  }
}

void drawCross(unsigned char *img, int s0, int s1, int s2, int drawchan, int c1, int c2, double rad)
{
  for (int i=c1-rad; i<c1+rad; i++)
    img[c2*s1*s0 + i*s0 + drawchan] = 255;  
  for (int i=c2-rad; i<c2+rad; i++)
    img[i*s1*s0 + c1*s0 + drawchan] = 255;  
}

void drawCrossWithColor(unsigned char *img, int s0, int s1, int s2, int c1, int c2, double rad, unsigned char *color)
{
  for (int k = 0; k<3; k++)
  {
    for (int i=c1-rad; i<c1+rad; i++)
      img[c2*s1*s0 + i*s0 + k] = color[k];  
    for (int i=c2-rad; i<c2+rad; i++)
      img[i*s1*s0 + c1*s0 + k] = color[k];  
  }
}

//draw the first N circles on the grid of RxC
void drawNCircle(unsigned char *img, int s0, int s1, int s2, int drawchan, int N, int g1, int g2)
{
  double rad;
  double w1 = s1/g1;
  double w2 = s2/g2;
  rad = w1<w2?w1/3:w2/3;
  for (int i=0; i<N; i++)
  {
    int gi1 = i/g1;
    int gi2 = i%g2;
    int pi1 = gi1*w1+w1/2;
    int pi2 = gi2*w2+w2/2;
    drawCircle(img,s0,s1,s2,drawchan,pi1,pi2,rad);
  }
}

double calDet44(double X[][4])
{
    double value = (
                    X[0][3]*X[1][2]*X[2][1]*X[3][0] - X[0][2]*X[1][3]*X[2][1]*X[3][0] - X[0][3]*X[1][1]*X[2][2]*X[3][0] + X[0][1]*X[1][3]*X[2][2]*X[3][0]+
                    X[0][2]*X[1][1]*X[2][3]*X[3][0] - X[0][1]*X[1][2]*X[2][3]*X[3][0] - X[0][3]*X[1][2]*X[2][0]*X[3][1] + X[0][2]*X[1][3]*X[2][0]*X[3][1]+
                    X[0][3]*X[1][0]*X[2][2]*X[3][1] - X[0][0]*X[1][3]*X[2][2]*X[3][1] - X[0][2]*X[1][0]*X[2][3]*X[3][1] + X[0][0]*X[1][2]*X[2][3]*X[3][1]+
                    X[0][3]*X[1][1]*X[2][0]*X[3][2] - X[0][1]*X[1][3]*X[2][0]*X[3][2] - X[0][3]*X[1][0]*X[2][1]*X[3][2] + X[0][0]*X[1][3]*X[2][1]*X[3][2]+
                    X[0][1]*X[1][0]*X[2][3]*X[3][2] - X[0][0]*X[1][1]*X[2][3]*X[3][2] - X[0][2]*X[1][1]*X[2][0]*X[3][3] + X[0][1]*X[1][2]*X[2][0]*X[3][3]+
                    X[0][2]*X[1][0]*X[2][1]*X[3][3] - X[0][0]*X[1][2]*X[2][1]*X[3][3] - X[0][1]*X[1][0]*X[2][2]*X[3][3] + X[0][0]*X[1][1]*X[2][2]*X[3][3]
                    );
    return value;
}

void invertMat44(double X[][4], double Y[][4])
{
    double det = calDet44(X);
    Y[0][0] = X[1][2]*X[2][3]*X[3][1] - X[1][3]*X[2][2]*X[3][1] + X[1][3]*X[2][1]*X[3][2] - X[1][1]*X[2][3]*X[3][2] - X[1][2]*X[2][1]*X[3][3] + X[1][1]*X[2][2]*X[3][3];
    Y[0][1] = X[0][3]*X[2][2]*X[3][1] - X[0][2]*X[2][3]*X[3][1] - X[0][3]*X[2][1]*X[3][2] + X[0][1]*X[2][3]*X[3][2] + X[0][2]*X[2][1]*X[3][3] - X[0][1]*X[2][2]*X[3][3];
    Y[0][2] = X[0][2]*X[1][3]*X[3][1] - X[0][3]*X[1][2]*X[3][1] + X[0][3]*X[1][1]*X[3][2] - X[0][1]*X[1][3]*X[3][2] - X[0][2]*X[1][1]*X[3][3] + X[0][1]*X[1][2]*X[3][3];
    Y[0][3] = X[0][3]*X[1][2]*X[2][1] - X[0][2]*X[1][3]*X[2][1] - X[0][3]*X[1][1]*X[2][2] + X[0][1]*X[1][3]*X[2][2] + X[0][2]*X[1][1]*X[2][3] - X[0][1]*X[1][2]*X[2][3];
    Y[1][0] = X[1][3]*X[2][2]*X[3][0] - X[1][2]*X[2][3]*X[3][0] - X[1][3]*X[2][0]*X[3][2] + X[1][0]*X[2][3]*X[3][2] + X[1][2]*X[2][0]*X[3][3] - X[1][0]*X[2][2]*X[3][3];
    Y[1][1] = X[0][2]*X[2][3]*X[3][0] - X[0][3]*X[2][2]*X[3][0] + X[0][3]*X[2][0]*X[3][2] - X[0][0]*X[2][3]*X[3][2] - X[0][2]*X[2][0]*X[3][3] + X[0][0]*X[2][2]*X[3][3];
    Y[1][2] = X[0][3]*X[1][2]*X[3][0] - X[0][2]*X[1][3]*X[3][0] - X[0][3]*X[1][0]*X[3][2] + X[0][0]*X[1][3]*X[3][2] + X[0][2]*X[1][0]*X[3][3] - X[0][0]*X[1][2]*X[3][3];
    Y[1][3] = X[0][2]*X[1][3]*X[2][0] - X[0][3]*X[1][2]*X[2][0] + X[0][3]*X[1][0]*X[2][2] - X[0][0]*X[1][3]*X[2][2] - X[0][2]*X[1][0]*X[2][3] + X[0][0]*X[1][2]*X[2][3];
    Y[2][0] = X[1][1]*X[2][3]*X[3][0] - X[1][3]*X[2][1]*X[3][0] + X[1][3]*X[2][0]*X[3][1] - X[1][0]*X[2][3]*X[3][1] - X[1][1]*X[2][0]*X[3][3] + X[1][0]*X[2][1]*X[3][3];
    Y[2][1] = X[0][3]*X[2][1]*X[3][0] - X[0][1]*X[2][3]*X[3][0] - X[0][3]*X[2][0]*X[3][1] + X[0][0]*X[2][3]*X[3][1] + X[0][1]*X[2][0]*X[3][3] - X[0][0]*X[2][1]*X[3][3];
    Y[2][2] = X[0][1]*X[1][3]*X[3][0] - X[0][3]*X[1][1]*X[3][0] + X[0][3]*X[1][0]*X[3][1] - X[0][0]*X[1][3]*X[3][1] - X[0][1]*X[1][0]*X[3][3] + X[0][0]*X[1][1]*X[3][3];
    Y[2][3] = X[0][3]*X[1][1]*X[2][0] - X[0][1]*X[1][3]*X[2][0] - X[0][3]*X[1][0]*X[2][1] + X[0][0]*X[1][3]*X[2][1] + X[0][1]*X[1][0]*X[2][3] - X[0][0]*X[1][1]*X[2][3];
    Y[3][0] = X[1][2]*X[2][1]*X[3][0] - X[1][1]*X[2][2]*X[3][0] - X[1][2]*X[2][0]*X[3][1] + X[1][0]*X[2][2]*X[3][1] + X[1][1]*X[2][0]*X[3][2] - X[1][0]*X[2][1]*X[3][2];
    Y[3][1] = X[0][1]*X[2][2]*X[3][0] - X[0][2]*X[2][1]*X[3][0] + X[0][2]*X[2][0]*X[3][1] - X[0][0]*X[2][2]*X[3][1] - X[0][1]*X[2][0]*X[3][2] + X[0][0]*X[2][1]*X[3][2];
    Y[3][2] = X[0][2]*X[1][1]*X[3][0] - X[0][1]*X[1][2]*X[3][0] - X[0][2]*X[1][0]*X[3][1] + X[0][0]*X[1][2]*X[3][1] + X[0][1]*X[1][0]*X[3][2] - X[0][0]*X[1][1]*X[3][2];
    Y[3][3] = X[0][1]*X[1][2]*X[2][0] - X[0][2]*X[1][1]*X[2][0] + X[0][2]*X[1][0]*X[2][1] - X[0][0]*X[1][2]*X[2][1] - X[0][1]*X[1][0]*X[2][2] + X[0][0]*X[1][1]*X[2][2];

    for (int i=0; i<4; i++)
        for (int j=0; j<4; j++)
            Y[i][j] = Y[i][j]/det;
}

void subtractVec(double *a, double *b, double *c, int s)
{
    for (int i=0; i<s; i++)
        c[i] = a[i]-b[i];
}

void negateVec(double *a, int s)
{
    for (int i=0; i<s; i++)
        a[i] = -a[i];
}

//s1,s2,s3: fastest to slowest
void sliceImageDouble(double *input, int s1, int s2, int s3, double *output, int indS1)
{
    for (int i=0; i<s3; i++)
        for (int j=0; j<s2; j++)
        {
            output[i*s2+j] = input[i*s2*s1+j*s1+indS1]*input[i*s2*s1+j*s1+s1-1];
        }
}

unsigned char quantizeDouble(double val, double minVal, double maxVal)
{
    return (val-minVal)*255.0/(maxVal-minVal);
}

//3D data, fastest to slowest
void quantizeImageDouble3D(double *input, unsigned char *output, int s0, int s1, int s2)
{
    double maxVal[4];
    maxVal[0] = maxVal[1] = maxVal[2] = maxVal[3] = -(1<<15);
    double minVal[4];
    minVal[0] = minVal[1] = minVal[2] = minVal[3] = ((1<<15) - 1);

    for (int i=0; i<s2; i++)
        for (int j=0; j<s1; j++)
            for (int k=0; k<s0; k++)
            {
                if (input[i*s1*s0+j*s0+k]>maxVal[k])
                    maxVal[k] = input[i*s1*s0+j*s0+k];
                if (input[i*s1*s0+j*s0+k]<minVal[k])
                    minVal[k] = input[i*s1*s0+j*s0+k];
            }
    for (int i=0; i<4; i++)
        printf("minmax %d = [%f,%f]\n",i,minVal[i],maxVal[i]);
    for (int i=0; i<s2; i++)
        for (int j=0; j<s1; j++)
            for (int k=0; k<s0; k++)
            {
                output[i*s1*s0+j*s0+k] = quantizeDouble(input[i*s1*s0+j*s0+k],minVal[k],maxVal[k]);
            }
}

template<class T>
void quantizeImage3D(T *input, unsigned char *output, int s0, int s1, int s2)
{
    double maxVal[4];
    maxVal[0] = maxVal[1] = maxVal[2] = maxVal[3] = -(1<<15);
    double minVal[4];
    minVal[0] = minVal[1] = minVal[2] = minVal[3] = ((1<<15) - 1);

    for (int i=0; i<s2; i++)
        for (int j=0; j<s1; j++)
            for (int k=0; k<s0; k++)
            {
                if (input[i*s1*s0+j*s0+k]>maxVal[k])
                    maxVal[k] = input[i*s1*s0+j*s0+k];
                if (input[i*s1*s0+j*s0+k]<minVal[k])
                    minVal[k] = input[i*s1*s0+j*s0+k];
            }
    for (int i=0; i<4; i++)
        printf("minmax %d = [%f,%f]\n",i,minVal[i],maxVal[i]);
    for (int i=0; i<s2; i++)
        for (int j=0; j<s1; j++)
            for (int k=0; k<s0; k++)
            {
                output[i*s1*s0+j*s0+k] = quantizeDouble(input[i*s1*s0+j*s0+k],minVal[k],maxVal[k]);
            }
}

void applyMask(unsigned char *input, int s0, int s1, int s2, int *mask, unsigned char *output)
{
    for (int i=0; i<s2; i++)
        for (int j=0; j<s1; j++)
            for (int k=0; k<s0; k++)
            {
                output[i*s1*s0+j*s0+k] = input[i*s1*s0+j*s0+k]*mask[i*s1+j];
            }
}

void removeChannel(unsigned char *input, int s0, int s1, int s2, int chan, unsigned char *output)
{
    memcpy(output,input,s0*s1*s2*sizeof(unsigned char));
    for (int i=0; i<s2; i++)
        for (int j=0; j<s1; j++)
                output[i*s1*s0+j*s0+chan] = 0;            
}
//---end of cuda_volume_rendering functions

template<class T>
void setPlane(T* image, int s1, int s2, int s3, T val, int s1i)
{
  for (int i=0; i<s3; i++)
    for (int j=0; j<s2; j++)
      image[i*s2*s1+j*s1+s1i] = val;
}

void transposeMat33(double X[][3], double Y[][3])
{
    for (int i=0; i<3; i++)
        for (int j=i; j<3; j++)
        {
            Y[i][j]=X[j][i];
            Y[j][i]=X[i][j];
        }
}

float linearizeDepth(float depth, float zNear, float zFar)
{
    return (2.0 * zFar * zNear) / (zFar + zNear - depth * (zFar - zNear));
}

float linearizeDepthOrtho(float depth, float zNear, float zFar)
{
    //the returned depth is relative to the "at" point
    return (depth*(zFar-zNear)+zFar+zNear)/2;
}



template<class T>
void saveImage(int width, int height, int nchan, T *data, char *name)
{
    TGAImage *img = new TGAImage(width,height);
    

    unsigned char* dataQuantized = new unsigned char[height*width*nchan];
    quantizeImage3D<T>(data,dataQuantized,nchan,width,height);

    Colour c;    
    for(int x=0; x<height; x++)
        for(int y=0; y<width; y++)
        {
            c.a = 255;
            c.b = c.g = c.r = 0;
            switch (nchan)
            {
              case 4:
                c.a = dataQuantized[x*width*nchan+y*nchan+3];
              case 3:
                c.b = dataQuantized[x*width*nchan+y*nchan+2];
              case 2:
                c.g = dataQuantized[x*width*nchan+y*nchan+1];
              case 1:
                c.r = dataQuantized[x*width*nchan+y*nchan];
            }                                        
            img->setPixel(c,x,y);
         }
    
    img->WriteImage(name);  
    delete img;
    delete[] dataQuantized;
}

template<class T>
void saveImageWithoutQuantizing(int width, int height, int nchan, T *data, char *name)
{
    TGAImage *img = new TGAImage(width,height);
    
    Colour c;    
    for(int x=0; x<height; x++)
        for(int y=0; y<width; y++)
        {
            c.a = 255;
            c.b = c.g = c.r = 0;
            switch (nchan)
            {
              case 4:
                c.a = data[x*width*nchan+y*nchan+3];
              case 3:
                c.b = data[x*width*nchan+y*nchan+2];
              case 2:
                c.g = data[x*width*nchan+y*nchan+1];
              case 1:
                c.r = data[x*width*nchan+y*nchan];
            }                                        
            img->setPixel(c,x,y);
        }
    
    img->WriteImage(name);  
    delete img;
}

//image1 and image2 should have same spatial size (except number of channels, i.e. fastest axis)
template <class T1, class T2>
void copyImageChannel(T1 *image1,int s10,int s11,int s12,int c1,T2 *image2,int s20,int c2)
{
  for (int i=0; i<s12; i++)
    for (int j=0; j<s11; j++)
    {
      int ind1 = i*s11*s10 + j*s10 + c1;
      int ind2 = i*s11*s20 + j*s20 + c2;
      image2[ind2] = image1[ind1];
    }
}

double computeAngle(double *v1, double *v2)
{
  double dp = dotProduct(v1,v2,3);
  return acos(dp)*180/M_PI;
}

void render(Hale::Viewer *viewer){
  viewer->draw();
  viewer->bufferSwap();
}

glm::vec4 convertDepthBuffToWorldPos(int w, int h, double depth, Hale::Viewer *viewer)
{
  double depthv = linearizeDepthOrtho(lerp(-1,1,0,depth,1),viewer->camera.clipNear(),viewer->camera.clipFar());
  double wv, hv;
  wv = w-viewer->widthBuffer()/2;
  hv = h-viewer->heightBuffer()/2;

  glm::vec3 diff = viewer->camera.at() - viewer->camera.from();
  double dist = glm::length(diff);
  double fangle = viewer->camera.fov()*AIR_PI/360;
  double vextent = dist*tan(fangle)*2;
  double pixelsize = vextent/viewer->heightBuffer();
  wv = wv*pixelsize;
  hv = hv*pixelsize;
  depthv = -(depthv+dist);
  printf("Inside convertDepthBuffToWorldPos: viewpos = %f,%f,%f\n",wv,hv,depthv);
  glm::vec4 vpos;
  vpos.x = wv;
  vpos.y = hv;
  vpos.z = depthv;
  vpos.w = 1;
  glm::vec4 result;
  result = viewer->camera.viewInv()*vpos;
  return result;
}

glm::vec4 convertDepthBuffToViewPos(int w, int h, double depth, Hale::Viewer *viewer)
{
  double depthv = linearizeDepthOrtho(lerp(-1,1,0,depth,1),viewer->camera.clipNear(),viewer->camera.clipFar());
  double wv, hv;
  wv = w-viewer->widthBuffer()/2;
  hv = h-viewer->heightBuffer()/2;

  glm::vec3 diff = viewer->camera.at() - viewer->camera.from();
  double dist = glm::length(diff);
  double fangle = viewer->camera.fov()*AIR_PI/360;
  double vextent = dist*tan(fangle)*2;
  double pixelsize = vextent/viewer->heightBuffer();
  wv = wv*pixelsize;
  hv = hv*pixelsize;
  depthv = -(depthv+dist);
  printf("Inside convertDepthBuffToViewPos: winpos = %d,%d; viewpos = %f,%f,%f\n",w,h,wv,hv,depthv);
  glm::vec4 vpos;
  vpos.x = wv;
  vpos.y = hv;
  vpos.z = depthv;
  vpos.w = 1;
  return vpos;
}

glm::vec4 convertWorldToViewPos(double x, double y, double z, Hale::Viewer *viewer)
{
  glm::vec4 wpos;
  wpos.x = x;
  wpos.y = y;
  wpos.z = z;
  wpos.w = 1;
  glm::vec4 vpos;
  vpos = viewer->camera.view()*wpos;
  return vpos;
}

void mainInit()
{
  for (int i=0; i<NTEX; i++)
    d_volumeArray[i] = 0;
}

class Queue
{
public:
  Queue(int isize)
  {
    size = isize;
    nin = nrrdNew();
    filemem0 = filemem1 = 0;
  }
  Queue()
  {
    size = NTEX;
    nin = nrrdNew();
    filemem0 = filemem1 = 0;
  }
  ~Queue()
  {

  }
  cudaArray* find(int time)
  {
    if (timetoindex.find(time) == timetoindex.end())
      return NULL;
    else
      return d_volumeArray[timetoindex[time]];
  }

  int findFarthestTime(int time)
  {
    int dismax = -1;
    int maxind;
    for (int i=0; i<elems.size(); i++)
      if (abs(time-elems[i])>dismax)
      {
        dismax = abs(time-elems[i]);
        maxind = i;
      }
    return maxind;
  }

  cudaArray* push(int time, int *arr_nameid, char* pathprefix, airArray *mop)
  {
    printf("Inside Queue.push(): time = %d\n",time);
    for (int i=0; i<elems.size(); i++)
      printf("%d ", elems[i]);
    printf("\n");
    cudaArray* findres = find(time);
    if (findres)
    {
      printf("findres = %d\n", findres);
      return findres;
    }
    int curvol;
    if (elems.size()<size)
    {
      elems.push_back(time);
      timetoindex[time] = elems.size()-1;
      curvol = elems.size()-1;
      printf("curvol in 'if': %d\n",curvol);
    }
    else
    {
      curvol = findFarthestTime(time);
      timetoindex.erase(elems[curvol]);
      elems[curvol] = time;
      timetoindex[time] = curvol;
      printf("curvol in 'else': %d\n",curvol);
    }

    char inname[1000];
    char *err;
    int curnameind = arr_nameid[time];
    sprintf(inname,"%s/%d.nrrd",pathprefix,curnameind);
    cout<<"inname = "<<inname<<endl;

    if (nrrdLoad(nin, inname, NULL)) {
      err = biffGetDone(NRRD);
      fprintf(stderr, "%s: trouble reading \"%s\":\n%s", "Queue.push()", inname, err);
      free(err);
      exit(1);
    }

    cout<<"read file "<<inname<<endl;
    unsigned int pixSize;
    cudaChannelFormatDesc channelDesc;
    pixSize = sizeof(float);
    channelDesc = cudaCreateChannelDesc<float>();

    if (3 != nin->dim && 3 != nin->spaceDim) {
        fprintf(stderr, "%s: need 3D array in 3D space, (not %uD in %uD)\n",
        "Queue.push()", nin->dim, nin->spaceDim);
        airMopError(mop); exit(1);
    }

    if (nin->dim == 3)
    {
        dim[0] = 1;
        dim[1] = nin->axis[0].size;
        dim[2] = nin->axis[1].size;
        dim[3] = nin->axis[2].size;
    }
    else //4-channel
    {
        dim[0] = nin->axis[0].size;
        dim[1] = nin->axis[1].size;
        dim[2] = nin->axis[2].size;
        dim[3] = nin->axis[3].size;
    }
    int channel = 1;

    if (!filemem0)
    {
      filemem0 = new float[dim[1]*dim[2]*dim[3]];
      filemem1 = new float[dim[1]*dim[2]*dim[3]];
    }

    for (int i=0; i<dim[1]*dim[2]*dim[3]; i++)
    {
        filemem0[i] = ((short*)nin->data)[i*2];
        filemem1[i] = ((short*)nin->data)[i*2+1];
    }

    const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);      
    
    if (!d_volumeArray[curvol])
    {
      cudaMalloc3DArray(&d_volumeArray[curvol], &channelDesc, volumeSize);
    }

    cudaMemcpy3DParms copyParams0 = {0};
    copyParams0.srcPtr   = make_cudaPitchedPtr((void*)filemem0, volumeSize.width*pixSize, volumeSize.width, volumeSize.height);
    copyParams0.dstArray = d_volumeArray[curvol];
    copyParams0.extent   = volumeSize;
    copyParams0.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams0);

    return d_volumeArray[curvol];
  }
  int* getDataDim()
  {
    return dim;
  }
private:
  int size;
  int cursize;
  vector<int> elems;
  unordered_map<int,int> timetoindex;
  //nin data
  float *filemem0, *filemem1;
  Nrrd *nin;
  int dim[4];
};

void interpolVolAndRender(int &curVolInMem, int mini, double alpha, Queue &queue, int *arr_nameid, 
  double *arr_center, char *pathprefix, airArray *mop, unsigned int pixSize, int *dim, int *size, 
  double *eigenvec, double verextent2, double swidth, double sstep, int nOutChannel, float *d_volmem, 
  int *d_dim, int *d_size, double *d_dir1, double *d_dir2, double *d_center, double *imageDouble, double *d_imageDouble,
  unsigned char *imageQuantized, Hale::Viewer &viewer, Hale::Viewer &viewer2, Hale::Polydata *hpldview2, 
  Hale::Polydata *hpld_inter, double spherescale)
{
  int count;
  double dir1[3],dir2[3],center[3];
  const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);
  cudaChannelFormatDesc channelDesc;
  channelDesc = cudaCreateChannelDesc<float>();

  if (curVolInMem != mini)
  {
    curVolInMem = mini;
    count = mini;
    cudaError_t errCu;
    cudaArray* d_curvolarr = queue.push(count,arr_nameid,pathprefix,mop);

    tex0.normalized = false;                    
    tex0.filterMode = cudaFilterModeLinear;     
    tex0.addressMode[0] = cudaAddressModeBorder;
    tex0.addressMode[1] = cudaAddressModeBorder;
    tex0.addressMode[2] = cudaAddressModeBorder;
    cudaBindTextureToArray(tex0, d_curvolarr, channelDesc);    

    count = mini+1;
    d_curvolarr = queue.push(count,arr_nameid,pathprefix,mop);
    tex1.normalized = false;                    
    tex1.filterMode = cudaFilterModeLinear;     
    tex1.addressMode[0] = cudaAddressModeBorder;
    tex1.addressMode[1] = cudaAddressModeBorder;
    tex1.addressMode[2] = cudaAddressModeBorder;
    cudaBindTextureToArray(tex1, d_curvolarr, channelDesc); 
  }   

  int numThread1D;
  
  numThread1D = 8;
  dim3 threadsPerBlock(numThread1D,numThread1D,numThread1D);
  dim3 numBlocks((dim[1]+numThread1D-1)/numThread1D,(dim[2]+numThread1D-1)/numThread1D,(dim[3]+numThread1D-1)/numThread1D);

  kernel_interpol<<<numBlocks,threadsPerBlock>>>(d_volmem,d_dim,alpha);

  cudaError_t errCu = cudaGetLastError();
  if (errCu != cudaSuccess) 
      printf("Error After kernel_nterpol when clicking: %s\n", cudaGetErrorString(errCu));

  errCu = cudaDeviceSynchronize();
  if (errCu != cudaSuccess) 
      printf("Error Sync After kernel_nterpol when clicking: %s\n", cudaGetErrorString(errCu));

  //copy from device's global mem to texture mem
  cudaMemcpy3DParms copyParams0 = {0};
  copyParams0.srcPtr   = make_cudaPitchedPtr((void*)d_volmem, volumeSize.width*pixSize, volumeSize.width, volumeSize.height);
  copyParams0.dstArray = d_volumeArray[NTEX];
  copyParams0.extent   = volumeSize;
  copyParams0.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&copyParams0);

  tex2.normalized = false;                    
  tex2.filterMode = cudaFilterModeLinear;     
  tex2.addressMode[0] = cudaAddressModeBorder;
  tex2.addressMode[1] = cudaAddressModeBorder;
  tex2.addressMode[2] = cudaAddressModeBorder;
  cudaBindTextureToArray(tex2, d_volumeArray[NTEX], channelDesc);       
  
  //after that call the normal kernel to do MIP
  count = mini;
  for (int i=0; i<3; i++)
    center[i] = cubicFilter<double>(alpha, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);
  
  double FT[3];
  double FN[3],FB[3];
  double dr[3],ddr[3];
  for (int i=0; i<3; i++)
    dr[i] = cubicFilter_G<double>(alpha, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);

  for (int i=0; i<3; i++)
    ddr[i] = cubicFilter_GG<double>(alpha, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);

  normalize(dr,3);
  normalize(ddr,3);

  memcpy(FT,dr,sizeof(double)*3);
  memcpy(FN,eigenvec,sizeof(double)*3);
  normalize(FN,3);
  cross(FT,FN,FB);
  cross(FB,FT,FN);
  memcpy(dir1,FN,sizeof(double)*3);
  memcpy(dir2,FB,sizeof(double)*3);

  cudaMemcpy(d_dir1, dir1, 3*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dir2, dir2, 3*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_center,center,3*sizeof(double), cudaMemcpyHostToDevice);

  numThread1D = 16;
  dim3 threadsPerBlock2(numThread1D,numThread1D);
  dim3 numBlocks2((size[0]+numThread1D-1)/numThread1D,(size[1]+numThread1D-1)/numThread1D);

  kernel_cpr<<<numBlocks2,threadsPerBlock2>>>(d_dim, d_size, verextent2, d_center, d_dir1, d_dir2, swidth, sstep, nOutChannel, d_imageDouble);

  errCu = cudaGetLastError();
  if (errCu != cudaSuccess) 
      printf("Error After kernel_cpr when clicking: %s\n", cudaGetErrorString(errCu));

  errCu = cudaDeviceSynchronize();
  if (errCu != cudaSuccess) 
      printf("Error Sync After kernel_cpr when clicking: %s\n", cudaGetErrorString(errCu));

  cudaMemcpy(imageDouble, d_imageDouble, sizeof(double)*size[0]*size[1]*nOutChannel, cudaMemcpyDeviceToHost);

  short width = size[0];
  short height = size[1];
  
  quantizeImageDouble3D(imageDouble,imageQuantized,4,size[0],size[1]);    
  setPlane<unsigned char>(imageQuantized, 4, size[0], size[1], 255, 3);

  viewer2.current();
  hpldview2->replaceLastTexture((unsigned char *)imageQuantized,size[0],size[1],4);
  
  viewer.current();
  glm::mat4 fmat2 = glm::mat4();
    
  fmat2[0][0] = spherescale;
  fmat2[1][1] = spherescale;
  fmat2[2][2] = spherescale;
  fmat2[3][0] = center[0];
  fmat2[3][1] = center[1];
  fmat2[3][2] = center[2];
  hpld_inter->model(fmat2);  
}

int
main(int argc, const char **argv) {
  const char *me;
  char *err;
  hestOpt *hopt=NULL;
  hestParm *hparm;
  airArray *mop;

  //cache queue for GPU memory
  //int queue[NTEX];
  //int queueCurSize = 0;
  //unordered_map<int,int> timetoindex;
  Queue queue;

  char *name;
  char *texname1, *texname2;
  
  double dir1[3],dir2[3];

  //tmp fixed track coords, and radius
  double track[3] = {366.653991263,89.6381792864,104.736646409};
  double trackhomo[4];
  trackhomo[0] = track[0];
  trackhomo[1] = track[1];
  trackhomo[2] = track[2];
  trackhomo[3] = 1;
  double trackw[4];
  double radius = 10;

  double center[3];

  int size[2];
  Nrrd *nin;
  char *outname;
  char inname[100];
  char *centername;
  double swidth, sstep; //width and step to take inside the slice  
  short *outdata;
  char outnameslice[100];
  double verextent; //vertical extent to project MIP
  char *pathprefix;
  int curVolInMem;

  /* boilerplate hest code */
  me = argv[0];
  mop = airMopNew();
  hparm = hestParmNew();
  airMopAdd(mop, hparm, (airMopper)hestParmFree, airMopAlways);
  /* setting up the command-line options */
  hparm->respFileEnable = AIR_TRUE;
  hparm->noArgsIsNoProblem = AIR_TRUE;

  hestOptAdd(&hopt, "isize", "sx sy", airTypeInt, 2, 2, size, "200 200",
             "output image sizes");

  hestOptAdd(&hopt, "vex", "ve", airTypeDouble, 1, 1, &verextent, "200",
             "vertical extent in projecting MIP");

  hestOptAdd(&hopt, "dir1", "x y z", airTypeDouble, 3, 3, dir1, "1 0 0",
             "first direction of the generated image");

  hestOptAdd(&hopt, "dir2", "x y z", airTypeDouble, 3, 3, dir2, "0 -1 0",
             "second direction of the generated image");

  hestOptAdd(&hopt, "swidth", "sw", airTypeDouble, 1, 1, &swidth, "1",
             "the width of the slice to cut");

  hestOptAdd(&hopt, "sstep", "ss", airTypeDouble, 1, 1, &sstep, "1",
             "the step of Maximum Intensity Projection through slice");  

  hestOptAdd(&hopt, "i", "name", airTypeString, 1, 1, &centername, "coord_newtrack_pioneer.txt", "name of files centaining centers");
  hestOptAdd(&hopt, "pref", "path", airTypeString, 1, 1, &pathprefix, "/media/trihuynh/781B8CE3469A7908/scivisdata", "prefix of the path to the folder containing data files");

  hestOptAdd(&hopt, "o", "name", airTypeString, 1, 1, &outname, "cpr.nrrd", "name of output image");

  hestParseOrDie(hopt, argc-1, argv+1, hparm,
                 me, "demo program", AIR_TRUE, AIR_TRUE, AIR_TRUE);
  airMopAdd(mop, hopt, (airMopper)hestOptFree, airMopAlways);
  airMopAdd(mop, hopt, (airMopper)hestParseFree, airMopAlways);

  /* Compute threshold (isovalue) */
  cout<<"After TEEM processing of input arguments"<<endl;

  int countline = 0;
  string line;
  ifstream infile(centername);
  int *arr_nameid;
  double *arr_center;

  while (std::getline(infile, line))
  {
    ++countline;    
  }

  infile.clear();
  infile.seekg(0, ios::beg);

  arr_nameid = new int[countline];
  arr_center = new double[countline*3];
  for (int i=0; i<countline; i++)
  {
    infile >> arr_nameid[i];
    infile >> arr_center[i*3];
    infile >> arr_center[i*3+1];
    infile >> arr_center[i*3+2];
  }
  infile.close();
  cout<<"Initialized countline = "<<countline<<endl;

  double thresdis = 1.0;
  vector<double> vcenter;
  vector<int> vnameid;

  vcenter.push_back(arr_center[0]);
  vcenter.push_back(arr_center[1]);
  vcenter.push_back(arr_center[2]);
  vnameid.push_back(arr_nameid[0]);

  double thresang = 150;
  //correction by thresholding distance
  
  for (int i=1; i<countline; i++)
  {
    int countv = vcenter.size();
    if (diss2P(vcenter[countv-3],vcenter[countv-2],vcenter[countv-1],arr_center[i*3+0],arr_center[i*3+1],arr_center[i*3+2])<thresdis)
    {
      continue;
    }
    else
    {
      vcenter.push_back(arr_center[i*3+0]);
      vcenter.push_back(arr_center[i*3+1]);
      vcenter.push_back(arr_center[i*3+2]);
      vnameid.push_back(arr_nameid[i]);
    }
  }
  countline = vcenter.size()/3;
  memcpy(arr_center,vcenter.data(),sizeof(double)*countline*3);
  memcpy(arr_nameid,vnameid.data(),sizeof(int)*countline);
  

  //correction by thresholding angle
  vcenter.clear();
  vcenter.push_back(arr_center[0]);
  vcenter.push_back(arr_center[1]);
  vcenter.push_back(arr_center[2]);
  vcenter.push_back(arr_center[3]);
  vcenter.push_back(arr_center[4]);
  vcenter.push_back(arr_center[5]);

  vnameid.clear();
  vnameid.push_back(arr_nameid[0]);
  vnameid.push_back(arr_nameid[1]);

  double prevec[3];
  prevec[0] = arr_center[3]-arr_center[0];
  prevec[1] = arr_center[4]-arr_center[1];
  prevec[2] = arr_center[5]-arr_center[2];
  normalize(prevec,3);
  for (int i=2; i<countline; i++)
  {
    double curvec[3];
    curvec[0] = arr_center[i*3+0]-arr_center[(i-1)*3+0];
    curvec[1] = arr_center[i*3+1]-arr_center[(i-1)*3+1];
    curvec[2] = arr_center[i*3+2]-arr_center[(i-1)*3+2];
    normalize(curvec,3);
    double ang = computeAngle(prevec,curvec);
    if (ang>thresang)
      continue;
    memcpy(prevec,curvec,sizeof(double)*3);
    vcenter.push_back(arr_center[i*3+0]);
    vcenter.push_back(arr_center[i*3+1]);
    vcenter.push_back(arr_center[i*3+2]);
    vnameid.push_back(arr_nameid[i]);
  }

  //adding more vertices at the beginning and ending to have enough convolution points
  /*
  double firstpoint[3];
  firstpoint[0] = vcenter[0];
  firstpoint[1] = vcenter[1];
  firstpoint[2] = vcenter[2];
  int firstnameid = vnameid[0];
  double lastpoint[3];
  lastpoint[0] = vcenter[vcenter.size()-3];
  lastpoint[1] = vcenter[vcenter.size()-2];
  lastpoint[2] = vcenter[vcenter.size()-1];
  int lastnameid = vnameid[vnameid.size()-1];


  vcenter.insert(vcenter.begin(),firstpoint[2]);
  vcenter.insert(vcenter.begin(),firstpoint[1]);
  vcenter.insert(vcenter.begin(),firstpoint[0]);
  vnameid.insert(vnameid.begin(),firstnameid);

  vcenter.push_back(lastpoint[0]);
  vcenter.push_back(lastpoint[1]);
  vcenter.push_back(lastpoint[2]);
  vcenter.push_back(lastpoint[0]);
  vcenter.push_back(lastpoint[1]);
  vcenter.push_back(lastpoint[2]);
  vnameid.push_back(lastnameid);
  vnameid.push_back(lastnameid);
  */
  printf("after correcting input\n");
  countline = vcenter.size()/3;
  memcpy(arr_center,vcenter.data(),sizeof(double)*countline*3);
  memcpy(arr_nameid,vnameid.data(),sizeof(int)*countline);

  outdata = new short[size[0]*size[1]*countline];

  cout<<"Initialized outdata"<<endl;

  int curnameind;
  
  float* filemem0 = NULL;
  float* filemem1 = NULL;
  int initalized = 0;
  double *imageDouble = NULL;
  int *d_dim;
  double *d_dir1;
  double *d_dir2;
  double *d_imageDouble;
  int *d_size;
  double *d_center;
  int count = 0;

  nin = nrrdNew();

  Nrrd *ndblpng = nrrdNew();

  float camfr[3], camat[3], camup[3], camnc, camfc, camFOV;
  int camortho;
  unsigned int camsize[2];
  
  camfr[0] = arr_center[countline/2*3+0];
  camfr[1] = arr_center[countline/2*3+1];
  camfr[2] = arr_center[countline/2*3+2]-5;
  camat[0] = arr_center[countline/2*3+0];
  camat[1] = arr_center[countline/2*3+1];
  camat[2] = arr_center[countline/2*3+2];
  camup[0] = 1;
  camup[1] = 0;
  camup[2] = 0;
  camnc = -10;
  camfc = 10;
  camFOV = 170;
  camortho = 1;
  camsize[0] = 500;
  camsize[1] = 500;
  

  //debug clicking
  /*
  camfr[0] = 0;
  camfr[1] = 1;
  camfr[2] = -6;
  camat[0] = 0;
  camat[1] = 1;
  camat[2] = -1;
  camup[0] = 1;
  camup[1] = 0;
  camup[2] = 0;
  camnc = -10;
  camfc = 10;
  camFOV = 90;
  camortho = 1;
  camsize[0] = 500;
  camsize[1] = 500;  
  */
  Hale::init();
  //Hale::debugging = 1;
  Hale::Scene scene;
  /* then create viewer (in order to create the OpenGL context) */
  Hale::Viewer viewer(camsize[0], camsize[1], "Viewer1", &scene);
  viewer.lightDir(glm::vec3(-1.0f, 1.0f, 3.0f));
  viewer.camera.init(glm::vec3(camfr[0], camfr[1], camfr[2]),
                     glm::vec3(camat[0], camat[1], camat[2]),
                     glm::vec3(camup[0], camup[1], camup[2]),
                     camFOV, (float)camsize[0]/camsize[1],
                     camnc, camfc, camortho);
  viewer.refreshCB((Hale::ViewerRefresher)render);
  viewer.refreshData(&viewer);
  viewer.current();

  Hale::Scene scene2;
  Hale::Viewer viewer2(camsize[0], camsize[1], "Viewer2", &scene2);
  viewer2.lightDir(glm::vec3(-1.0f, 1.0f, 3.0f));
  viewer2.camera.init(glm::vec3(camfr[0], camfr[1], camfr[2]),
                     glm::vec3(camat[0], camat[1], camat[2]),
                     glm::vec3(camup[0], camup[1], camup[2]),
                     camFOV, (float)camsize[0]/camsize[1],
                     camnc, camfc, camortho);
  viewer2.refreshCB((Hale::ViewerRefresher)render);
  viewer2.refreshData(&viewer2);

  printf("Initialized viewer\n");

  Hale::Program *newprog = new Hale::Program("tex-vert-cpr.glsl","texdemo-frag.glsl");
  newprog->compile();
  newprog->bindAttribute(Hale::vertAttrIdxXYZW, "positionVA");
  newprog->bindAttribute(Hale::vertAttrIdxRGBA, "colorVA");
  newprog->bindAttribute(Hale::vertAttrIdxNorm, "normalVA");
  newprog->bindAttribute(Hale::vertAttrIdxTex2, "tex2VA");
  newprog->link();    

  double spherescale = 0.2;
  double spherescale_inter = 0.3;

  int density = 10; //how many points per one unit length in index-space
  int countls = 0;
  for (int i=1; i<countline-3; i++)
  {
    double dis = sqrt(diss2P(arr_center[i*3+0], arr_center[i*3+1], arr_center[i*3+2],
                            arr_center[(i+1)*3+0], arr_center[(i+1)*3+1], arr_center[(i+1)*3+2]));
    countls += (dis*density);
  }

  int *ptotime = new int[countls];
  double *ptofrac = new double[countls];
  int *timetop = new int[countline];
  memset(timetop,0,sizeof(int)*countline);
  limnPolyData *lpld3 = limnPolyDataNew();

  limnPolyDataAlloc(lpld3, 0, countls, countls, 1);

  int cpointind = 0;
  for (int i=1; i<countline-3; i++)
  {
    double dis = sqrt(diss2P(arr_center[i*3+0], arr_center[i*3+1], arr_center[i*3+2],
                            arr_center[(i+1)*3+0], arr_center[(i+1)*3+1], arr_center[(i+1)*3+2]));
    int countseg = dis*density;
    double tsep = 1.0/((double)countseg);
    timetop[i] = cpointind;
    for (int j=0; j<countseg; j++)
    {
      double curpoint[3];
      for (int k=0; k<3; k++)
        curpoint[k] = cubicFilter<double>((double)j*tsep, arr_center[(i-1)*3+k], arr_center[(i)*3+k], arr_center[(i+1)*3+k], arr_center[(i+2)*3+k]);
      ELL_4V_SET(lpld3->xyzw + 4*cpointind, curpoint[0],curpoint[1],curpoint[2], 1.0);  
      lpld3->indx[cpointind] = cpointind;

      ptotime[cpointind] = i;
      ptofrac[cpointind] = j*tsep;
      cpointind++;      
    }
  }  
  lpld3->type[0] = limnPrimitiveLineStrip;
  lpld3->icnt[0] = countls;

  printf("countls = %d\n", countls);

  //adding linestrip for original path
  limnPolyData *lpldorig = limnPolyDataNew();
  limnPolyDataAlloc(lpldorig, 0, countline-3, countline-3, 1);
  for (int i=1; i<countline-2; i++)
  {
    ELL_4V_SET(lpldorig->xyzw + 4*(i-1), arr_center[(i)*3+0],arr_center[(i)*3+1],arr_center[(i)*3+2], 1.0); 
    lpldorig->indx[i-1] = i-1;
  }
  lpldorig->type[0] = limnPrimitiveLineStrip;
  lpldorig->icnt[0] = countline-3;
  Hale::Polydata *hpldorig = new Hale::Polydata(lpldorig, true,
                         Hale::ProgramLib(Hale::preprogramAmbDiffSolid),
                         "LineStrip");  
  hpldorig->colorSolid(1.0,1.0,0.5);
  scene.add(hpldorig);

  limnPolyData *lpld4 = limnPolyDataNew();
  limnPolyDataSpiralTubeWrap(lpld4, lpld3,
                           0, NULL,
                           10, 4,
                           0.1);

  Hale::Polydata *hpld3 = new Hale::Polydata(lpld3, true,
                         Hale::ProgramLib(Hale::preprogramAmbDiffSolid),
                         "LineStrip");  
  hpld3->colorSolid(1.0,1.0,0.5);

  Hale::Polydata *hpld4 = new Hale::Polydata(lpld4, true,
                         Hale::ProgramLib(Hale::preprogramAmbDiffSolid),
                         "SpiralTube");
  hpld4->colorSolid(1.0,1.0,0.5);  
  scene.add(hpld4);

  vector<Hale::Polydata *> vtexture;
  vector<Hale::Polydata *> vtexture2;
  vector<Hale::Polydata *> vsphere;
  vector<Hale::Polydata *> vsphereorig;
  unsigned char *imageQuantized;
  imageQuantized = new unsigned char[size[0]*size[1]*4];
  double prevFT[3], prevFN[3], prevFB[3];

  printf("countline after adding boundary points = %d\n", countline);
  printf("arr_nameid[1] = %d\n", arr_nameid[1]);
  printf("arr_nameid[countline-3] = %d\n", arr_nameid[countline-3]);
  printf("New nameid and centers:\n");
  for (int i=0; i<countline; i++)
    printf("%d %f %f %f\n", arr_nameid[i], arr_center[i*3+0], arr_center[i*3+1], arr_center[i*3+2]);

  //computing PCA
  double cov[9];
  computeCovariance(arr_center,countline,cov);
  double eigval[3];
  eigenOfHess(cov,eigval);
  double seigval = eigval[0];
  for (int i=1; i<3; i++)
    if (seigval>eigval[i])
      seigval = eigval[i];
  double eigenvec[3];

  computeEigenVec(cov,seigval,eigenvec);
  printf("eigenvector is (%f,%f,%f)\n", eigenvec[0],eigenvec[1],eigenvec[2]);

  int nOutChannel = 4;

  for (count = 1; count<countline-2; count++)
  {
    curnameind = arr_nameid[count];
    for (int i=0; i<3; i++)
      center[i] = cubicFilter<double>(0, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);
    printf("center = %f %f %f\n", center[0],center[1],center[2]);
    
    double FT[3];
    double FN[3],FB[3];
    double dr[3],ddr[3];
    for (int i=0; i<3; i++)
      dr[i] = cubicFilter_G<double>(0, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);
    for (int i=0; i<3; i++)
      ddr[i] = cubicFilter_GG<double>(0, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);

    printf("dr = (%f,%f,%f)\n",dr[0],dr[1],dr[2]);
    printf("ddr = (%f,%f,%f)\n",ddr[0],ddr[1],ddr[2]);
    normalize(dr,3);
    normalize(ddr,3);
    printf("after normalizing\n");
    printf("dr = (%f,%f,%f)\n",dr[0],dr[1],dr[2]);
    printf("ddr = (%f,%f,%f)\n",ddr[0],ddr[1],ddr[2]);

    memcpy(FT,dr,sizeof(double)*3);
    //double crossddrdr[3];
    //cross(ddr,dr,crossddrdr);
    //cross(dr,crossddrdr,FN);
    memcpy(FN,eigenvec,sizeof(double)*3);
    normalize(FN,3);
    cross(FT,FN,FB);
    cross(FB,FT,FN);
    normalize(FN,3);
    normalize(FT,3);
    normalize(FB,3);
    memcpy(dir1,FN,sizeof(double)*3);
    memcpy(dir2,FB,sizeof(double)*3);

    printf("N = %f %f %f, B = %f %f %f, T = %f %f %f, dotNB = %f, dotNT = %f, dotBT = %f\n",FN[0],FN[1],FN[2],FB[0],FB[1],FB[2],FT[0],FT[1],FT[2],
      dotProduct(FN,FB,3),dotProduct(FN,FT,3),dotProduct(FB,FT,3));

    if (count>1)
    {
      printf("count = %d\n", count);
      printf("angle of FT: %f\n", computeAngle(FT,prevFT));
      printf("angle of FN: %f\n", computeAngle(FN,prevFN));
      printf("angle of FB: %f\n", computeAngle(FB,prevFB));
    }

    memcpy(prevFB,FB,sizeof(double)*3);
    memcpy(prevFN,FN,sizeof(double)*3);
    memcpy(prevFT,FT,sizeof(double)*3);

    limnPolyData *lpld = limnPolyDataNew();
    limnPolyDataSquare(lpld, 1 << limnPolyDataInfoNorm | 1 << limnPolyDataInfoTex2);

    printf("after initializing lpld\n");
    
    Hale::Polydata *hpld = new Hale::Polydata(lpld, true,
                         NULL,
                         "square");

    hpld->program(newprog);
    
    glm::mat4 tmat = glm::mat4();
    
    tmat[0][0] = FN[0];
    tmat[0][1] = FN[1];
    tmat[0][2] = FN[2];
    tmat[0][3] = 0;
    tmat[1][0] = FB[0];
    tmat[1][1] = FB[1];
    tmat[1][2] = FB[2];
    tmat[1][3] = 0;
    tmat[2][0] = FT[0];
    tmat[2][1] = FT[1];
    tmat[2][2] = FT[2];
    tmat[2][3] = 0;
    tmat[3][0] = center[0];
    tmat[3][1] = center[1];
    tmat[3][2] = center[2];
    tmat[3][3] = 1;
    
    glm::mat4 smat = glm::mat4();
    smat[0][0] = 2;
    smat[1][1] = 2;
    glm::mat4 fmat = tmat*smat;

    hpld->model(fmat);    
    
    //add a sphere
    limnPolyData *lpld2 = limnPolyDataNew();
    limnPolyDataIcoSphere(lpld2, 1 << limnPolyDataInfoNorm, 3);

    Hale::Polydata *hpld2 = new Hale::Polydata(lpld2, true,
                         Hale::ProgramLib(Hale::preprogramAmbDiffSolid),
                         "IcoSphere");
    hpld2->colorSolid(lerp(0,1,0,count,countline-1),lerp(1,0,0,count,countline-1),0.5);
    
    glm::mat4 fmat2 = glm::mat4();
    
    fmat2[0][0] = spherescale;
    fmat2[1][1] = spherescale;
    fmat2[2][2] = spherescale;
    fmat2[3][0] = center[0];
    fmat2[3][1] = center[1];
    fmat2[3][2] = center[2];
    fmat2[3][3] = 1;    

    hpld2->model(fmat2);    

    scene.add(hpld2);    
    vsphere.push_back(hpld2);

    //adding sphere for original track path too
    limnPolyData *lpldorigsp = limnPolyDataNew();
    limnPolyDataIcoSphere(lpldorigsp, 1 << limnPolyDataInfoNorm, 3);

    Hale::Polydata *hpldorigsp = new Hale::Polydata(lpldorigsp, true,
                         Hale::ProgramLib(Hale::preprogramAmbDiffSolid),
                         "IcoSphere");
    hpldorigsp->colorSolid(lerp(0,1,0,count,countline-1),lerp(1,0,0,count,countline-1),0.5);
    
    fmat2[0][0] = spherescale;
    fmat2[1][1] = spherescale;
    fmat2[2][2] = spherescale;
    fmat2[3][0] = arr_center[(count)*3+0];
    fmat2[3][1] = arr_center[(count)*3+1];
    fmat2[3][2] = arr_center[(count)*3+2];
    fmat2[3][3] = 1;

    hpldorigsp->model(fmat2);    
    scene.add(hpldorigsp);

    vsphereorig.push_back(hpldorigsp);
    
    printf("after adding hpld to scene\n");

    printf("added lpld\n");

    cout<<"Before read in file, with curnameind = "<<curnameind<<", center = "<<center[0]<<" "<<center[1]<<" "<<center[2]<<endl;
/*
    sprintf(inname,"%s/%d.nrrd",pathprefix,curnameind);
    cout<<"inname = "<<inname<<endl;

    if (nrrdLoad(nin, inname, NULL)) {
      err = biffGetDone(NRRD);
      fprintf(stderr, "%s: trouble reading \"%s\":\n%s", me, inname, err);
      free(err);
      return;
    }

    cout<<"read file "<<inname<<endl;
    unsigned int pixSize;
    cudaChannelFormatDesc channelDesc;
    pixSize = sizeof(float);
    channelDesc = cudaCreateChannelDesc<float>();

    if (3 != nin->dim && 3 != nin->spaceDim) {
        fprintf(stderr, "%s: need 3D array in 3D space, (not %uD in %uD)\n",
        argv[0], nin->dim, nin->spaceDim);
        airMopError(mop); exit(1);
    }

    double mat_trans[4][4];

    mat_trans[3][0] = mat_trans[3][1] = mat_trans[3][2] = 0;
    mat_trans[3][3] = 1;

    int dim[4];
    if (nin->dim == 3)
    {
        dim[0] = 1;
        dim[1] = nin->axis[0].size;
        dim[2] = nin->axis[1].size;
        dim[3] = nin->axis[2].size;
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {                
                mat_trans[j][i] = nin->axis[i].spaceDirection[j];
            }
            mat_trans[i][3] = nin->spaceOrigin[i];
        }
    }
    else //4-channel
    {
        dim[0] = nin->axis[0].size;
        dim[1] = nin->axis[1].size;
        dim[2] = nin->axis[2].size;
        dim[3] = nin->axis[3].size;
        for (int i=0; i<3; i++) {
            for (int j=0; j<3; j++) {
                mat_trans[j][i] = nin->axis[i+1].spaceDirection[j];
            }
            mat_trans[i][3] = nin->spaceOrigin[i];
        }
    }
    int channel = 1;

    if (!initalized)
    {
      filemem0 = new float[dim[1]*dim[2]*dim[3]];
      filemem1 = new float[dim[1]*dim[2]*dim[3]];
    }

    for (int i=0; i<dim[1]*dim[2]*dim[3]; i++)
    {
        filemem0[i] = ((short*)nin->data)[i*2];
        filemem1[i] = ((short*)nin->data)[i*2+1];
    }

    double mat_trans_inv[4][4];
    invertMat44(mat_trans,mat_trans_inv);
   //tex3D stuff
    const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);

    printf("Array size: %f MB\n", dim[1]*dim[2]*dim[3]*sizeof(float)/1024.0/1024.0);
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t errCu;
    errCu = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != errCu ){
      printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(errCu) );
      exit(1);
    }
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage (before copying memory to Device): used = %f MB, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    
    
    if (!initalized)
    {
      cudaMalloc3DArray(&d_volumeArray[2], &channelDesc, volumeSize);
    }

    cudaMemcpy3DParms copyParams0 = {0};
    copyParams0.srcPtr   = make_cudaPitchedPtr((void*)filemem0, volumeSize.width*pixSize, volumeSize.width, volumeSize.height);
    copyParams0.dstArray = d_volumeArray[2];
    copyParams0.extent   = volumeSize;
    copyParams0.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams0);
*/

    size_t free_byte;
    size_t total_byte;
    double free_db;
    double total_db;
    double used_db;

    cudaError_t errCu;
    cudaChannelFormatDesc channelDesc;
    channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* d_curvolarr = queue.push(count,arr_nameid,pathprefix,mop);
    tex2.normalized = false;                      // access with normalized texture coordinates
    tex2.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex2.addressMode[0] = cudaAddressModeBorder;   // wrap texture coordinates
    tex2.addressMode[1] = cudaAddressModeBorder;
    tex2.addressMode[2] = cudaAddressModeBorder;
    cudaBindTextureToArray(tex2, d_curvolarr, channelDesc);

    errCu = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if ( cudaSuccess != errCu ){
      printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(errCu) );
      exit(1);
    }

    free_db = (double)free_byte ;
    total_db = (double)total_byte ;
    used_db = total_db - free_db ;
    printf("GPU memory usage (after copying memory to Device): used = %f MB, free = %f MB, total = %f MB\n",
            used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    

    nOutChannel = 4;
    //int dim[4];
    //memcpy(dim,queue.getDataDim(),sizeof(int)*4);
    //printf("dim = %d,%d,%d,%d\n",dim[0],dim[1],dim[2],dim[3]);

    if (!initalized)
    {
      imageDouble = new double[size[0]*size[1]*nOutChannel];

      errCu = cudaMalloc(&d_dim, 4*sizeof(int));
      if ( cudaSuccess != errCu ){
          printf("Error in Malloc of d_dim: %s \n", cudaGetErrorString(errCu) );
          exit(1);
      }
      errCu = cudaMemcpy(d_dim, queue.getDataDim(), 4*sizeof(int), cudaMemcpyHostToDevice);
      if ( cudaSuccess != errCu ){
          printf("Error in memcpy of d_dim: %s \n", cudaGetErrorString(errCu) );
          exit(1);
      }
      errCu = cudaMalloc(&d_dir1, sizeof(dir1));
      if ( cudaSuccess != errCu ){
          printf("Error in Malloc or memcpy: %s \n", cudaGetErrorString(errCu) );
          exit(1);
      }      
      errCu = cudaMalloc(&d_dir2, sizeof(dir2));      
      if ( cudaSuccess != errCu ){
          printf("Error in Malloc or memcpy: %s \n", cudaGetErrorString(errCu) );
          exit(1);
      }
      errCu = cudaMalloc(&d_imageDouble,sizeof(double)*size[0]*size[1]*nOutChannel);
      if ( cudaSuccess != errCu ){
          printf("Error in Malloc or memcpy: %s \n", cudaGetErrorString(errCu) );
          exit(1);
      }
      errCu = cudaMalloc(&d_size,2*sizeof(int));
      if ( cudaSuccess != errCu ){
          printf("Error in Malloc or memcpy: %s \n", cudaGetErrorString(errCu) );
          exit(1);
      }      
      errCu = cudaMemcpy(d_size,size,2*sizeof(int), cudaMemcpyHostToDevice);
      if ( cudaSuccess != errCu ){
          printf("Error in Malloc or memcpy: %s \n", cudaGetErrorString(errCu) );
          exit(1);
      }
      errCu = cudaMalloc(&d_center,3*sizeof(double));
      if ( cudaSuccess != errCu ){
          printf("Error in Malloc or memcpy: %s \n", cudaGetErrorString(errCu) );
          exit(1);
      }      
    }

    errCu = cudaMemcpy(d_dir1, dir1, 3*sizeof(double), cudaMemcpyHostToDevice);
      if ( cudaSuccess != errCu ){
          printf("Error in Malloc or memcpy: %s \n", cudaGetErrorString(errCu) );
          exit(1);
      }    
    errCu = cudaMemcpy(d_dir2, dir2, 3*sizeof(double), cudaMemcpyHostToDevice);
      if ( cudaSuccess != errCu ){
          printf("Error in Malloc or memcpy: %s \n", cudaGetErrorString(errCu) );
          exit(1);
      }    
    errCu = cudaMemcpy(d_center,center,3*sizeof(double), cudaMemcpyHostToDevice);
      if ( cudaSuccess != errCu ){
          printf("Error in Malloc or memcpy: %s \n", cudaGetErrorString(errCu) );
          exit(1);
      }

    int numThread1D = 16;
    dim3 threadsPerBlock(numThread1D,numThread1D);
    dim3 numBlocks((size[0]+numThread1D-1)/numThread1D,(size[1]+numThread1D-1)/numThread1D);

    kernel_cpr<<<numBlocks,threadsPerBlock>>>(d_dim, d_size, verextent, d_center, d_dir1, d_dir2, swidth, sstep, nOutChannel, d_imageDouble);

    errCu = cudaGetLastError();
    if (errCu != cudaSuccess) 
        printf("Error After first kernel_cpr: %s\n", cudaGetErrorString(errCu));

    errCu = cudaDeviceSynchronize();
    if (errCu != cudaSuccess) 
        printf("Error Sync: %s\n", cudaGetErrorString(errCu));

    cudaMemcpy(imageDouble, d_imageDouble, sizeof(double)*size[0]*size[1]*nOutChannel, cudaMemcpyDeviceToHost);

    short width = size[0];
    short height = size[1];

    copyImageChannel<double,short>(imageDouble,4,size[0],size[1],1,outdata+count*size[0]*size[1],1,0);
    
    quantizeImageDouble3D(imageDouble,imageQuantized,4,size[0],size[1]);    
    setPlane<unsigned char>(imageQuantized, 4, size[0], size[1], 255, 3);
    drawNCircle(imageQuantized,4,size[0],size[1],0, count, countline/2,countline/2);

    hpld->setTexture((char*)"myTextureSampler",(unsigned char *)imageQuantized,size[0],size[1],4);
    scene.add(hpld);
    vtexture.push_back(hpld);
    
    drawCircle(imageQuantized,4,size[0],size[1],0,size[0]/2,size[1]/2,20);
    double trackedcenter[3];
    trackedcenter[0] = arr_center[count*3];
    trackedcenter[1] = arr_center[count*3+1];
    trackedcenter[2] = arr_center[count*3+2];
    double centerdiff[3];
    subtractVec(trackedcenter,center,centerdiff,3);
    double coorfn, coorfb, coorft;
    coorfn = dotProduct(centerdiff,FN,3);
    coorfb = dotProduct(centerdiff,FB,3);
    coorft = dotProduct(centerdiff,FT,3);
    unsigned char color[3];
    color[0] = color[1] = color[2] = 128;
    if (coorft<-swidth/2 || coorft>swidth/2)
    {
      color[0] = color[1] = color[2] = 0;
    }
    else
    {
      color[0] = lerp(255,0,-swidth/2,coorft,swidth/2);
      color[1] = lerp(0,0,-swidth/2,coorft,swidth/2);
      color[2] = lerp(0,255,-swidth/2,coorft,swidth/2);
    }
    drawCrossWithColor(imageQuantized,4,size[0],size[1],size[0]/2+coorfn*size[1]/verextent,size[1]/2+coorfb*size[1]/verextent,20,color);

    initalized = 1;
    sprintf(outnameslice,"cpr_seq_%d.png",curnameind);
    if (nrrdWrap_va(ndblpng, imageQuantized, nrrdTypeUChar, 3, 4, width, height)
      || nrrdSave(outnameslice, ndblpng, NULL)
          ) {
      char *err = biffGetDone(NRRD);
      printf("%s: couldn't save output:\n%s", argv[0], err);
      free(err); nrrdNix(ndblpng);
      exit(1);
      }
  }

  cout<<"Before allocating output nrrd"<<endl;  
  Nrrd *ndbl = nrrdNew();

  cout<<"Before saving output nrrd"<<endl;
  if (nrrdWrap_va(ndbl, outdata, nrrdTypeShort, 3, size[0], size[1], countline)
        || nrrdSave(outname,ndbl,NULL)
        ) 
  {
    char *err = biffGetDone(NRRD);
    printf("%s: couldn't save output:\n%s", argv[0], err);
    free(err); nrrdNix(ndbl);
    exit(1);
  }

  viewer2.current();  
  printf("after setting viewer2.current()\n");
    
    limnPolyData *lpldview2 = limnPolyDataNew();
    limnPolyDataSquare(lpldview2, 1 << limnPolyDataInfoNorm | 1 << limnPolyDataInfoTex2);

    Hale::Polydata *hpldview2 = new Hale::Polydata(lpldview2, true,
                         NULL,//Hale::ProgramLib(Hale::preprogramAmbDiffSolid),
                         "square");
    Hale::Program *newprog2 = new Hale::Program("texdemo-vert.glsl","texdemo-frag.glsl");
    newprog2->compile();
    newprog2->bindAttribute(Hale::vertAttrIdxXYZW, "positionVA");
    newprog2->bindAttribute(Hale::vertAttrIdxRGBA, "colorVA");
    newprog2->bindAttribute(Hale::vertAttrIdxNorm, "normalVA");
    newprog2->bindAttribute(Hale::vertAttrIdxTex2, "tex2VA");
    newprog2->link();  

    hpldview2->program(newprog2);

    //find lerping between 2 volumes
    count = 1;
    /*
    curnameind = arr_nameid[count];
    sprintf(inname,"%s/%d.nrrd",pathprefix,curnameind);
    if (nrrdLoad(nin, inname, NULL)) {
      err = biffGetDone(NRRD);
      fprintf(stderr, "%s: trouble reading \"%s\":\n%s", me, inname, err);
      free(err);
      return;
    }

    cout<<"read file "<<inname<<endl;
    unsigned int pixSize;
    cudaChannelFormatDesc channelDesc;
    pixSize = sizeof(float);
    channelDesc = cudaCreateChannelDesc<float>();

    if (3 != nin->dim && 3 != nin->spaceDim) {
        fprintf(stderr, "%s: need 3D array in 3D space, (not %uD in %uD)\n",
        argv[0], nin->dim, nin->spaceDim);
        airMopError(mop); exit(1);
    }

    int dim[4];
    if (nin->dim == 3)
    {
        dim[0] = 1;
        dim[1] = nin->axis[0].size;
        dim[2] = nin->axis[1].size;
        dim[3] = nin->axis[2].size;
    }
    else //4-channel
    {
        dim[0] = nin->axis[0].size;
        dim[1] = nin->axis[1].size;
        dim[2] = nin->axis[2].size;
        dim[3] = nin->axis[3].size;
    }
    int channel = 1;

    if (!filemem0)
    {
      filemem0 = new float[dim[1]*dim[2]*dim[3]];
      filemem1 = new float[dim[1]*dim[2]*dim[3]];
    }

    for (int i=0; i<dim[1]*dim[2]*dim[3]; i++)
    {
        filemem0[i] = ((short*)nin->data)[i*2];
        filemem1[i] = ((short*)nin->data)[i*2+1];
    }

    //debug
    for (int k=0; k<=2; k++)
      for (int j=0; j<=2; j++)
        for (int i=0; i<=2; i++)
          printf("volume 1: at (%d,%d,%d) = %f\n", i,j,k,filemem0[k*dim[1]*dim[2]+j*dim[1]+i]);

    const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);

    if (!d_volumeArray[0])
      cudaMalloc3DArray(&d_volumeArray[0], &channelDesc, volumeSize);
    
    cudaMemcpy3DParms copyParams0 = {0};
    copyParams0.srcPtr   = make_cudaPitchedPtr((void*)filemem0, volumeSize.width*pixSize, volumeSize.width, volumeSize.height);
    copyParams0.dstArray = d_volumeArray[0];
    copyParams0.extent   = volumeSize;
    copyParams0.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams0);
    */
    cudaChannelFormatDesc channelDesc;
    channelDesc = cudaCreateChannelDesc<float>();    
    cudaArray* d_curvolarr = queue.push(count,arr_nameid,pathprefix,mop);

    tex0.normalized = false;                    
    tex0.filterMode = cudaFilterModeLinear;     
    tex0.addressMode[0] = cudaAddressModeBorder;
    tex0.addressMode[1] = cudaAddressModeBorder;
    tex0.addressMode[2] = cudaAddressModeBorder;
    cudaBindTextureToArray(tex0, d_curvolarr, channelDesc);    

    //read second file
    count = 2;
    /*
    curnameind = arr_nameid[count];
    sprintf(inname,"%s/%d.nrrd",pathprefix,curnameind);
    if (nrrdLoad(nin, inname, NULL)) {
      err = biffGetDone(NRRD);
      fprintf(stderr, "%s: trouble reading \"%s\":\n%s", me, inname, err);
      free(err);
      return;
    }

    cout<<"read file "<<inname<<endl;

    if (3 != nin->dim && 3 != nin->spaceDim) {
        fprintf(stderr, "%s: need 3D array in 3D space, (not %uD in %uD)\n",
        argv[0], nin->dim, nin->spaceDim);
        airMopError(mop); exit(1);
    }

    for (int i=0; i<dim[1]*dim[2]*dim[3]; i++)
    {
        filemem0[i] = ((short*)nin->data)[i*2];
        filemem1[i] = ((short*)nin->data)[i*2+1];
    }

    if (!d_volumeArray[1])
      cudaMalloc3DArray(&d_volumeArray[1], &channelDesc, volumeSize);

    cudaMemcpy3DParms copyParams1 = {0};
    copyParams1.srcPtr   = make_cudaPitchedPtr((void*)filemem0, volumeSize.width*pixSize, volumeSize.width, volumeSize.height);
    copyParams1.dstArray = d_volumeArray[1];
    copyParams1.extent   = volumeSize;
    copyParams1.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams1);
*/
    d_curvolarr = queue.push(count,arr_nameid,pathprefix,mop);
    tex1.normalized = false;                    
    tex1.filterMode = cudaFilterModeLinear;     
    tex1.addressMode[0] = cudaAddressModeBorder;
    tex1.addressMode[1] = cudaAddressModeBorder;
    tex1.addressMode[2] = cudaAddressModeBorder;
    cudaBindTextureToArray(tex1, d_curvolarr, channelDesc);    
    int curinterp = (timetop[1]+timetop[2])/2;
    double alpha = ptofrac[curinterp];
    cudaError_t errCu;

    float *d_volmem;
    int *dim = queue.getDataDim();
    const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);
    int pixSize = sizeof(float);
    cudaMalloc(&d_volmem,sizeof(float)*dim[1]*dim[2]*dim[3]);

    int numThread1D = 8;
    dim3 threadsPerBlock(numThread1D,numThread1D,numThread1D);
    dim3 numBlocks((dim[1]+numThread1D-1)/numThread1D,(dim[2]+numThread1D-1)/numThread1D,(dim[3]+numThread1D-1)/numThread1D);

    
    kernel_interpol<<<numBlocks,threadsPerBlock>>>(d_volmem,d_dim,alpha);

    errCu = cudaGetLastError();
    if (errCu != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(errCu));

    errCu = cudaDeviceSynchronize();
    if (errCu != cudaSuccess) 
        printf("Error Sync: %s\n", cudaGetErrorString(errCu));
    cudaMemcpy3DParms copyParams0 = {0};
    copyParams0.srcPtr   = make_cudaPitchedPtr((void*)d_volmem, volumeSize.width*pixSize, volumeSize.width, volumeSize.height);
    copyParams0.dstArray = d_volumeArray[NTEX];
    copyParams0.extent   = volumeSize;
    copyParams0.kind     = cudaMemcpyDeviceToDevice;
    cudaMemcpy3D(&copyParams0);

    //cudaArray* d_curvolarr = queue.push(count,arr_nameid,pathprefix,mop);
    tex2.normalized = false;                    
    tex2.filterMode = cudaFilterModeLinear;     
    tex2.addressMode[0] = cudaAddressModeBorder;
    tex2.addressMode[1] = cudaAddressModeBorder;
    tex2.addressMode[2] = cudaAddressModeBorder;
    cudaBindTextureToArray(tex2, d_volumeArray[NTEX], channelDesc);       
  
    count = 1;
    for (int i=0; i<3; i++)
      center[i] = cubicFilter<double>(alpha, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);

    printf("center = %f %f %f\n", center[0],center[1],center[2]);
    
    double FT[3];
    double FN[3],FB[3];
    double dr[3],ddr[3];
    for (int i=0; i<3; i++)
      dr[i] = cubicFilter_G<double>(alpha, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);

    for (int i=0; i<3; i++)
      ddr[i] = cubicFilter_GG<double>(alpha, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);

    printf("dr = (%f,%f,%f)\n",dr[0],dr[1],dr[2]);
    printf("ddr = (%f,%f,%f)\n",ddr[0],ddr[1],ddr[2]);
    normalize(dr,3);
    normalize(ddr,3);
    printf("after normalizing\n");
    printf("dr = (%f,%f,%f)\n",dr[0],dr[1],dr[2]);
    printf("ddr = (%f,%f,%f)\n",ddr[0],ddr[1],ddr[2]);

    memcpy(FT,dr,sizeof(double)*3);
    memcpy(FN,eigenvec,sizeof(double)*3);
    normalize(FN,3);
    cross(FT,FN,FB);
    cross(FB,FT,FN);
    memcpy(dir1,FN,sizeof(double)*3);
    memcpy(dir2,FB,sizeof(double)*3);
    printf("Interpolation: N = %f %f %f, B = %f %f %f, T = %f %f %f, dotNB = %f, dotNT = %f, dotBT = %f\n",FN[0],FN[1],FN[2],FB[0],FB[1],FB[2],FT[0],FT[1],FT[2],
      dotProduct(FN,FB,3),dotProduct(FN,FT,3),dotProduct(FB,FT,3));

    cudaMemcpy(d_dir1, dir1, 3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dir2, dir2, 3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_center,center,3*sizeof(double), cudaMemcpyHostToDevice);

    numThread1D = 16;
    dim3 threadsPerBlock2(numThread1D,numThread1D);
    dim3 numBlocks2((size[0]+numThread1D-1)/numThread1D,(size[1]+numThread1D-1)/numThread1D);

    kernel_cpr<<<numBlocks2,threadsPerBlock2>>>(d_dim, d_size, verextent, d_center, d_dir1, d_dir2, swidth, sstep, nOutChannel, d_imageDouble);

    errCu = cudaGetLastError();
    if (errCu != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(errCu));

    errCu = cudaDeviceSynchronize();
    if (errCu != cudaSuccess) 
        printf("Error Sync: %s\n", cudaGetErrorString(errCu));

    cudaMemcpy(imageDouble, d_imageDouble, sizeof(double)*size[0]*size[1]*nOutChannel, cudaMemcpyDeviceToHost);

    short width = size[0];
    short height = size[1];
    
    quantizeImageDouble3D(imageDouble,imageQuantized,4,size[0],size[1]);    
    setPlane<unsigned char>(imageQuantized, 4, size[0], size[1], 255, 3);

    hpldview2->setTexture((char*)"myTextureSampler",(unsigned char *)imageQuantized,size[0],size[1],4);

    //add a sphere for the interpolated position
    viewer.current();
    limnPolyData *lpld2 = limnPolyDataNew();
    limnPolyDataIcoSphere(lpld2, 1 << limnPolyDataInfoNorm, 3);

    Hale::Polydata *hpld_inter = new Hale::Polydata(lpld2, true,
                         Hale::ProgramLib(Hale::preprogramAmbDiffSolid),
                         "IcoSphere");
    hpld_inter->colorSolid(0,0,1.0);
    
    glm::mat4 fmat2 = glm::mat4();
    
    fmat2[0][0] = spherescale_inter;
    fmat2[1][1] = spherescale_inter;
    fmat2[2][2] = spherescale_inter;
    fmat2[3][0] = center[0];
    fmat2[3][1] = center[1];
    fmat2[3][2] = center[2];
    fmat2[3][3] = 1;
    printf("center of the first interpolated point: %f,%f,%f\n", center[0],center[1],center[2]);

    hpld_inter->model(fmat2);    

    scene.add(hpld_inter);   
    viewer2.current();

  curVolInMem = 1;
    
  scene2.add(hpldview2);
  //scene2.add(hpld3);
  scene2.drawInit();
  printf("after adding to scene2 and drawInit()\n");
  viewer2.verbose(3);

  render(&viewer2);
  printf("after rendering viewer2\n");
  viewer.current();
  viewer.verbose(3);

  //add a testing sphere
  /*
    viewer.current();
    limnPolyData *lpldtestsphere = limnPolyDataNew();
    limnPolyDataIcoSphere(lpldtestsphere, 1 << limnPolyDataInfoNorm, 3);

    Hale::Polydata *hpldtestsphere = new Hale::Polydata(lpldtestsphere, true,
                         Hale::ProgramLib(Hale::preprogramAmbDiffSolid),
                         "IcoSphere");
    hpldtestsphere->colorSolid(1.0,0,0);
    
    
    glm::mat4 fmat3 = glm::mat4();
    
    fmat3[0][0] = 1;
    fmat3[1][1] = 1;
    fmat3[2][2] = 1;
    fmat3[3][0] = 4;
    fmat3[3][1] = 5;
    fmat3[3][2] = 5;
    fmat3[3][3] = 1;    

    hpldtestsphere->model(fmat3);    
    

    scene.add(hpldtestsphere);    
  */
  /////////////////////////////////


  

  cout<<"After saving output nrrd"<<endl;
  scene.drawInit();
  printf("after scene.drawInit()\n");
  render(&viewer);
  
  printf("after render(&viewer)\n");

  //getting Z-buffer
  printf("viewer: width = %d, height = %d, widthBuffer = %d, heightBuffer = %d\n", viewer.width(),viewer.height(),viewer.widthBuffer(),viewer.heightBuffer());
  GLfloat* zbuffer = new GLfloat[viewer.widthBuffer()*viewer.heightBuffer()];

  glReadPixels(0,0,viewer.widthBuffer(),viewer.heightBuffer(),GL_DEPTH_COMPONENT,GL_FLOAT,zbuffer);  
  printf("Z-buffer\n");

  int wposw = 56;
  int hposw = 62;
  double dposw = zbuffer[hposw*viewer.widthBuffer()+wposw];
  printf("Before converting: wpos = %d, hpos = %d, dpos = %f\n", wposw,hposw,dposw);
  glm::vec4 wposworld = convertDepthBuffToWorldPos(wposw,hposw,dposw,&viewer);
  printf("After converting: x,y,z = %f,%f,%f\n", wposworld.x,wposworld.y,wposworld.z);
  glm::vec4 vpostest,wpostest;
  wpostest.x = arr_center[3];
  wpostest.y = arr_center[4];
  wpostest.z = arr_center[5];
  wpostest.w = 1;
  vpostest = viewer.camera.view()*wpostest;
  printf("World Pos Test = %f,%f,%f; View Pos Test = %f,%f,%f\n", wpostest.x,wpostest.y,wpostest.z,vpostest.x,vpostest.y,vpostest.z);

  float minz=1000,maxz=-1000;
  for (int i=0; i<viewer.widthBuffer()*viewer.heightBuffer(); i++)
  {
    zbuffer[i] = linearizeDepthOrtho(lerp(-1,1,0,zbuffer[i],1),viewer.camera.clipNear(),viewer.camera.clipFar());
    if (zbuffer[i]<minz)
        minz = zbuffer[i];
    if (zbuffer[i]>maxz)
        maxz = zbuffer[i];
  }
  printf("minmaxz = (%f,%f)\n",minz,maxz );
  saveImage<GLfloat>(viewer.widthBuffer(),viewer.heightBuffer(),1,zbuffer,"depth.tga");

  bool stateBKey = false;
  bool stateMKey = false;
  bool stateNKey = false;
  bool stateZoom = false;
  bool stateXKey = false;
  double lastX, lastY;
  double verextent2 = verextent;
  bool isHoldOn = false;
  bool checkPath = false;
  int stateBKeyInt = 0;
  while(!Hale::finishing){
    glfwWaitEvents();
    int keyPressed = viewer.getKeyPressed();
    if (keyPressed == GLFW_KEY_LEFT)
    {      
      if (curinterp>0)
      {
        curinterp--;
        interpolVolAndRender(curVolInMem, ptotime[curinterp], ptofrac[curinterp], queue, arr_nameid, 
                            arr_center, pathprefix, mop, pixSize, dim, size, 
                            eigenvec, verextent2, swidth, sstep, nOutChannel, d_volmem, 
                            d_dim, d_size, d_dir1, d_dir2, d_center, imageDouble, d_imageDouble,
                            imageQuantized, viewer, viewer2, hpldview2, 
                            hpld_inter, spherescale_inter);        
      }      
    }
    else
      if (keyPressed == GLFW_KEY_RIGHT)
      {
        if (curinterp<countls-1)
        {
          curinterp++;
          interpolVolAndRender(curVolInMem, ptotime[curinterp], ptofrac[curinterp], queue, arr_nameid, 
                              arr_center, pathprefix, mop, pixSize, dim, size, 
                              eigenvec, verextent2, swidth, sstep, nOutChannel, d_volmem, 
                              d_dim, d_size, d_dir1, d_dir2, d_center, imageDouble, d_imageDouble,
                              imageQuantized, viewer, viewer2, hpldview2, 
                              hpld_inter, spherescale_inter);        
        }      
      }

    if (stateBKey!=viewer.getStateBKey())
    {
      stateBKey = viewer.getStateBKey();
      stateBKeyInt = (stateBKeyInt+1)%3;
      if (stateBKeyInt == 1)
      {
        scene.remove(hpld4);
        scene.add(hpld3);
      }
      else if (stateBKeyInt == 2)
      {
        scene.remove(hpld3);
      }
      else
      {
        //scene.remove(hpld3);
        scene.add(hpld4);
      }
    }
    if (keyPressed == 'M')
    {
      stateMKey = !stateMKey;
      if (stateMKey)
      {
        for (int i=0; i<vtexture.size(); i++)
          scene.remove(vtexture[i]);
      }
      else
      {
        for (int i=0; i<vtexture.size(); i++)
          scene.add(vtexture[i]);
      }
    }
    if (keyPressed == 'N')
    {
      stateNKey = !stateNKey;
      if (stateNKey)
      {
        for (int i=0; i<vsphereorig.size(); i++)
          scene.remove(vsphereorig[i]);
        scene.remove(hpldorig);
      }
      else
      {
        for (int i=0; i<vsphereorig.size(); i++)
          scene.add(vsphereorig[i]);
        scene.add(hpldorig);
      }
    }
    if (keyPressed == 'X')
    {
      stateXKey = !stateXKey;
      if (stateXKey)
      {
        for (int i=0; i<vsphere.size(); i++)
          scene.remove(vsphere[i]);
      }
      else
      {
        for (int i=0; i<vsphere.size(); i++)
          scene.add(vsphere[i]);
      }
    }

    //processing zooming in the second window (MIP image)
    if (stateZoom)
    {
      if (!viewer2.getButton(0))
      {
        /*
        double curY = viewer2.getLastY();
        int heightBuff = viewer2.heightBuffer();
        double pcent = (curY-lastY)/heightBuff;
        printf("percent zoom = %f (curY = %f, lastY = %f, heightBuff = %d)\n", pcent, curY,lastY,heightBuff);
        stateZoom = false;

        verextent2 = verextent2*(1+pcent);

        kernel_cpr<<<numBlocks2,threadsPerBlock2>>>(d_dim, d_size, verextent2, d_center, d_dir1, d_dir2, swidth, sstep, nOutChannel, d_imageDouble);

        errCu = cudaGetLastError();
        if (errCu != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(errCu));

        errCu = cudaDeviceSynchronize();
        if (errCu != cudaSuccess) 
            printf("Error Sync: %s\n", cudaGetErrorString(errCu));

        cudaMemcpy(imageDouble, d_imageDouble, sizeof(double)*size[0]*size[1]*nOutChannel, cudaMemcpyDeviceToHost);
        
        quantizeImageDouble3D(imageDouble,imageQuantized,4,size[0],size[1]);    
        setPlane<unsigned char>(imageQuantized, 4, size[0], size[1], 255, 3);

        viewer2.current();
        hpldview2->replaceLastTexture((unsigned char *)imageQuantized,size[0],size[1],4);
        */
        stateZoom = false;
      }
      else
        if (std::isnan(lastY))
          lastY = viewer2.getLastY();
        else
        {
          double curY = viewer2.getLastY();
          int heightBuff = viewer2.heightBuffer();
          double pcent = (curY-lastY)/heightBuff;
          printf("percent zoom = %f (curY = %f, lastY = %f, heightBuff = %d)\n", pcent, curY,lastY,heightBuff);
          //stateZoom = false;

          verextent2 = verextent2*(1+pcent);

          kernel_cpr<<<numBlocks2,threadsPerBlock2>>>(d_dim, d_size, verextent2, d_center, d_dir1, d_dir2, swidth, sstep, nOutChannel, d_imageDouble);

          errCu = cudaGetLastError();
          if (errCu != cudaSuccess) 
              printf("Error: %s\n", cudaGetErrorString(errCu));

          errCu = cudaDeviceSynchronize();
          if (errCu != cudaSuccess) 
              printf("Error Sync: %s\n", cudaGetErrorString(errCu));

          cudaMemcpy(imageDouble, d_imageDouble, sizeof(double)*size[0]*size[1]*nOutChannel, cudaMemcpyDeviceToHost);
          
          quantizeImageDouble3D(imageDouble,imageQuantized,4,size[0],size[1]);    
          setPlane<unsigned char>(imageQuantized, 4, size[0], size[1], 255, 3);

          viewer2.current();
          hpldview2->replaceLastTexture((unsigned char *)imageQuantized,size[0],size[1],4);

          lastY = curY;
        }
    }
    else
      if (viewer2.getButton(0) && viewer2.getMode()==Hale::viewerModeZoom)
      {
        stateZoom = true;
        lastY = viewer2.getLastY();
        printf("Begin zooming: lastY = %f\n",lastY);
      }

    if (viewer.getButton(0))
    {      
      if (!isHoldOn)
      {
        GLfloat* zbufferC = new GLfloat[viewer.widthBuffer()*viewer.heightBuffer()];
        glReadPixels(0,0,viewer.widthBuffer(),viewer.heightBuffer(),GL_DEPTH_COMPONENT,GL_FLOAT,zbufferC);  
        int wposwC = viewer.getClickedX();
        int hposwC = viewer.heightBuffer()-viewer.getClickedY();
        double dposwC = zbufferC[hposwC*viewer.widthBuffer()+wposwC];
        printf("First Clicked (w,h,depth) = %d,%d,%f\n", wposwC,hposwC,dposwC);
        glm::vec4 wposviewC = convertDepthBuffToViewPos(wposwC,hposwC,dposwC,&viewer);
        printf("First Clicked View Pos = %f,%f,%f\n", wposviewC.x,wposviewC.y,wposviewC.z);
        //debug
        //glm::vec4 testpostview = convertWorldToViewPos(4,5,4,&viewer);      
        //printf("View Pos of World Pos (4,5,4) is = %f,%f,%f\n", testpostview.x,testpostview.y,testpostview.z);
        ////
        isHoldOn = true;
        checkPath = (dposwC<1.0);
        viewer.setPaused(checkPath);
      }
      
      if (checkPath)
      {
        GLfloat* zbuffer = new GLfloat[viewer.widthBuffer()*viewer.heightBuffer()];
        glReadPixels(0,0,viewer.widthBuffer(),viewer.heightBuffer(),GL_DEPTH_COMPONENT,GL_FLOAT,zbuffer);  
        int wposw = viewer.getLastX();
        int hposw = viewer.heightBuffer()-viewer.getLastY();
        double dposw = zbuffer[hposw*viewer.widthBuffer()+wposw];
        printf("Drag Clicked (w,h,depth) = %d,%d,%f\n", wposw,hposw,dposw);
        glm::vec4 wposview = convertDepthBuffToViewPos(wposw,hposw,dposw,&viewer);
        printf("Drag Clicked View Pos = %f,%f,%f\n", wposview.x,wposview.y,wposview.z);
        if (dposw<1.0)
        {
          double dismin = INT_MAX;
          int mini = -1;
          for (int i=1; i<countline-3; i++)
          {
            glm::vec4 curposview = convertWorldToViewPos(arr_center[i*3+0],arr_center[i*3+1],arr_center[i*3+2],&viewer);          
            //test using on x and y in view coordinate to find the closest point (not using z)
            //double dis1 = diss2P(wposview.x,wposview.y,wposview.z,curposview.x,curposview.y,curposview.z);
            glm::vec4 curposview2 = convertWorldToViewPos(arr_center[(i+1)*3+0],arr_center[(i+1)*3+1],arr_center[(i+1)*3+2],&viewer);
            double vec1[3],vec2[3];
            vec1[0] = curposview.x-wposview.x;
            vec1[1] = curposview.y-wposview.y;
            vec1[2] = 0;
            vec2[0] = curposview2.x-wposview.x;
            vec2[1] = curposview2.y-wposview.y;
            vec2[2] = 0;
            normalize(vec1,3);
            normalize(vec2,3);
            double angle = computeAngle(vec1,vec2);
            printf("Point %d (%d), View pos 1 = %f,%f,%f; View pos 2 = %f,%f,%f; angle = %f\n",i,arr_nameid[i],curposview.x,curposview.y,curposview.z,curposview2.x,curposview2.y,curposview2.z, angle);
            if (angle > 100)
            {
              double dis1 = diss2P(wposview.x,wposview.y,0,curposview.x,curposview.y,0);
              //double dis2 = diss2P(wposview.x,wposview.y,wposview.z,curposview.x,curposview.y,curposview.z);
              double dis2 = diss2P(wposview.x,wposview.y,0,curposview2.x,curposview2.y,0);
              printf("Point %d (%d) View Pos = %f,%f,%f, dis = %f\n",i,arr_nameid[i], curposview.x,curposview.y,curposview.z,dis1);
              if (dis1+dis2<dismin)
              {
                dismin = dis1+dis2;
                mini = i;
              }
            }        
          }
          //int numsample = 20;
          dismin = INT_MAX;
          double mint = -1;
          /*
          for (int i=0; i<=numsample; i++)
          {
            double t = (double)i/(double)numsample;
            double center[3];
            for (int j=0; j<3; j++)
              center[j] = cubicFilter<double>(t, arr_center[(mini-1)*3+j], arr_center[(mini)*3+j], arr_center[(mini+1)*3+j], arr_center[(mini+2)*3+j]);
            glm::vec4 curposview = convertWorldToViewPos(center[0],center[1],center[2],&viewer);
            //double dis = diss2P(wposview.x,wposview.y,wposview.z,curposview.x,curposview.y,curposview.z);
            double dis = diss2P(wposview.x,wposview.y,1,curposview.x,curposview.y,1);

            if (dis<dismin)
            {
              dismin = dis;
              mint = t;
            }
          }
          */
          int curp = timetop[mini];
          while (curp<countls && ptotime[curp]==mini)
          {
            glm::vec4 curposview = convertWorldToViewPos(lpld3->xyzw[curp*4+0],lpld3->xyzw[curp*4+1],lpld3->xyzw[curp*4+2],&viewer);
            double dis = diss2P(wposview.x,wposview.y,1,curposview.x,curposview.y,1);

            if (dis<dismin)
            {
              dismin = dis;
              mint = ptofrac[curp];
              curinterp = curp;
            }            
            curp++;
          }

          if (curVolInMem != mini)
          {
            curVolInMem = mini;
            //find lerping between 2 volumes
            count = mini;
            /*
            curnameind = arr_nameid[count];
            sprintf(inname,"%s/%d.nrrd",pathprefix,curnameind);
            if (nrrdLoad(nin, inname, NULL)) {
              err = biffGetDone(NRRD);
              fprintf(stderr, "%s: trouble reading \"%s\":\n%s", me, inname, err);
              free(err);
              return;
            }

            cout<<"read file "<<inname<<endl;
            unsigned int pixSize;
            cudaChannelFormatDesc channelDesc;
            pixSize = sizeof(float);
            channelDesc = cudaCreateChannelDesc<float>();

            if (3 != nin->dim && 3 != nin->spaceDim) {
                fprintf(stderr, "%s: need 3D array in 3D space, (not %uD in %uD)\n",
                argv[0], nin->dim, nin->spaceDim);
                airMopError(mop); exit(1);
            }

            int dim[4];
            if (nin->dim == 3)
            {
                dim[0] = 1;
                dim[1] = nin->axis[0].size;
                dim[2] = nin->axis[1].size;
                dim[3] = nin->axis[2].size;
            }
            else //4-channel
            {
                dim[0] = nin->axis[0].size;
                dim[1] = nin->axis[1].size;
                dim[2] = nin->axis[2].size;
                dim[3] = nin->axis[3].size;
            }
            int channel = 1;

            for (int i=0; i<dim[1]*dim[2]*dim[3]; i++)
            {
                filemem0[i] = ((short*)nin->data)[i*2];
                filemem1[i] = ((short*)nin->data)[i*2+1];
            }

            //debug
            for (int k=0; k<=2; k++)
              for (int j=0; j<=2; j++)
                for (int i=0; i<=2; i++)
                  printf("volume 1: at (%d,%d,%d) = %f\n", i,j,k,filemem0[k*dim[1]*dim[2]+j*dim[1]+i]);

            const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);

            if (!d_volumeArray[0])
              cudaMalloc3DArray(&d_volumeArray[0], &channelDesc, volumeSize);

            cudaMemcpy3DParms copyParams0 = {0};
            copyParams0.srcPtr   = make_cudaPitchedPtr((void*)filemem0, volumeSize.width*pixSize, volumeSize.width, volumeSize.height);
            copyParams0.dstArray = d_volumeArray[0];
            copyParams0.extent   = volumeSize;
            copyParams0.kind     = cudaMemcpyHostToDevice;
            cudaMemcpy3D(&copyParams0);
            */
            cudaError_t errCu;
            cudaChannelFormatDesc channelDesc;
            channelDesc = cudaCreateChannelDesc<float>();
            cudaArray* d_curvolarr = queue.push(count,arr_nameid,pathprefix,mop);

            tex0.normalized = false;                    
            tex0.filterMode = cudaFilterModeLinear;     
            tex0.addressMode[0] = cudaAddressModeBorder;
            tex0.addressMode[1] = cudaAddressModeBorder;
            tex0.addressMode[2] = cudaAddressModeBorder;
            cudaBindTextureToArray(tex0, d_curvolarr, channelDesc);    

            //read second file
            count = mini+1;
            /*
            curnameind = arr_nameid[count];
            sprintf(inname,"%s/%d.nrrd",pathprefix,curnameind);
            if (nrrdLoad(nin, inname, NULL)) {
              err = biffGetDone(NRRD);
              fprintf(stderr, "%s: trouble reading \"%s\":\n%s", me, inname, err);
              free(err);
              return;
            }

            cout<<"read file "<<inname<<endl;

            if (3 != nin->dim && 3 != nin->spaceDim) {
                fprintf(stderr, "%s: need 3D array in 3D space, (not %uD in %uD)\n",
                argv[0], nin->dim, nin->spaceDim);
                airMopError(mop); exit(1);
            }

            for (int i=0; i<dim[1]*dim[2]*dim[3]; i++)
            {
                filemem0[i] = ((short*)nin->data)[i*2];
                filemem1[i] = ((short*)nin->data)[i*2+1];
            }

            if (!d_volumeArray[1])
              cudaMalloc3DArray(&d_volumeArray[1], &channelDesc, volumeSize);

            cudaMemcpy3DParms copyParams1 = {0};
            copyParams1.srcPtr   = make_cudaPitchedPtr((void*)filemem0, volumeSize.width*pixSize, volumeSize.width, volumeSize.height);
            copyParams1.dstArray = d_volumeArray[1];
            copyParams1.extent   = volumeSize;
            copyParams1.kind     = cudaMemcpyHostToDevice;
            cudaMemcpy3D(&copyParams1);
            */
            d_curvolarr = queue.push(count,arr_nameid,pathprefix,mop);
            tex1.normalized = false;                    
            tex1.filterMode = cudaFilterModeLinear;     
            tex1.addressMode[0] = cudaAddressModeBorder;
            tex1.addressMode[1] = cudaAddressModeBorder;
            tex1.addressMode[2] = cudaAddressModeBorder;
            cudaBindTextureToArray(tex1, d_curvolarr, channelDesc); 
          }   

          int numThread1D;
          alpha = mint;
          
          numThread1D = 8;
          dim3 threadsPerBlock(numThread1D,numThread1D,numThread1D);
          dim3 numBlocks((dim[1]+numThread1D-1)/numThread1D,(dim[2]+numThread1D-1)/numThread1D,(dim[3]+numThread1D-1)/numThread1D);

          double alpha = mint;
          kernel_interpol<<<numBlocks,threadsPerBlock>>>(d_volmem,d_dim,alpha);

          cudaError_t errCu = cudaGetLastError();
          if (errCu != cudaSuccess) 
              printf("Error After kernel_nterpol when clicking: %s\n", cudaGetErrorString(errCu));

          errCu = cudaDeviceSynchronize();
          if (errCu != cudaSuccess) 
              printf("Error Sync After kernel_nterpol when clicking: %s\n", cudaGetErrorString(errCu));

          //copy from device's global mem to texture mem
          copyParams0.srcPtr   = make_cudaPitchedPtr((void*)d_volmem, volumeSize.width*pixSize, volumeSize.width, volumeSize.height);
          copyParams0.dstArray = d_volumeArray[NTEX];
          copyParams0.extent   = volumeSize;
          copyParams0.kind     = cudaMemcpyDeviceToDevice;
          cudaMemcpy3D(&copyParams0);

          tex2.normalized = false;                    
          tex2.filterMode = cudaFilterModeLinear;     
          tex2.addressMode[0] = cudaAddressModeBorder;
          tex2.addressMode[1] = cudaAddressModeBorder;
          tex2.addressMode[2] = cudaAddressModeBorder;
          cudaBindTextureToArray(tex2, d_volumeArray[NTEX], channelDesc);       
          
          //after that call the normal kernel to do MIP
          count = mini;
          for (int i=0; i<3; i++)
            center[i] = cubicFilter<double>(alpha, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);

          printf("center = %f %f %f\n", center[0],center[1],center[2]);
          
          double FT[3];
          double FN[3],FB[3];
          double dr[3],ddr[3];
          for (int i=0; i<3; i++)
            dr[i] = cubicFilter_G<double>(alpha, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);

          for (int i=0; i<3; i++)
            ddr[i] = cubicFilter_GG<double>(alpha, arr_center[(count-1)*3+i], arr_center[(count)*3+i], arr_center[(count+1)*3+i], arr_center[(count+2)*3+i]);

          printf("dr = (%f,%f,%f)\n",dr[0],dr[1],dr[2]);
          printf("ddr = (%f,%f,%f)\n",ddr[0],ddr[1],ddr[2]);
          normalize(dr,3);
          normalize(ddr,3);
          printf("after normalizing\n");
          printf("dr = (%f,%f,%f)\n",dr[0],dr[1],dr[2]);
          printf("ddr = (%f,%f,%f)\n",ddr[0],ddr[1],ddr[2]);

          memcpy(FT,dr,sizeof(double)*3);
          memcpy(FN,eigenvec,sizeof(double)*3);
          normalize(FN,3);
          cross(FT,FN,FB);
          cross(FB,FT,FN);
          memcpy(dir1,FN,sizeof(double)*3);
          memcpy(dir2,FB,sizeof(double)*3);
          printf("Interpolation: N = %f %f %f, B = %f %f %f, T = %f %f %f, dotNB = %f, dotNT = %f, dotBT = %f\n",FN[0],FN[1],FN[2],FB[0],FB[1],FB[2],FT[0],FT[1],FT[2],
            dotProduct(FN,FB,3),dotProduct(FN,FT,3),dotProduct(FB,FT,3));

          cudaMemcpy(d_dir1, dir1, 3*sizeof(double), cudaMemcpyHostToDevice);
          cudaMemcpy(d_dir2, dir2, 3*sizeof(double), cudaMemcpyHostToDevice);
          cudaMemcpy(d_center,center,3*sizeof(double), cudaMemcpyHostToDevice);

          numThread1D = 16;
          dim3 threadsPerBlock2(numThread1D,numThread1D);
          dim3 numBlocks2((size[0]+numThread1D-1)/numThread1D,(size[1]+numThread1D-1)/numThread1D);

          kernel_cpr<<<numBlocks2,threadsPerBlock2>>>(d_dim, d_size, verextent2, d_center, d_dir1, d_dir2, swidth, sstep, nOutChannel, d_imageDouble);

          errCu = cudaGetLastError();
          if (errCu != cudaSuccess) 
              printf("Error After kernel_cpr when clicking: %s\n", cudaGetErrorString(errCu));

          errCu = cudaDeviceSynchronize();
          if (errCu != cudaSuccess) 
              printf("Error Sync After kernel_cpr when clicking: %s\n", cudaGetErrorString(errCu));

          cudaMemcpy(imageDouble, d_imageDouble, sizeof(double)*size[0]*size[1]*nOutChannel, cudaMemcpyDeviceToHost);

          short width = size[0];
          short height = size[1];
          
          quantizeImageDouble3D(imageDouble,imageQuantized,4,size[0],size[1]);    
          setPlane<unsigned char>(imageQuantized, 4, size[0], size[1], 255, 3);

          viewer2.current();
          hpldview2->replaceLastTexture((unsigned char *)imageQuantized,size[0],size[1],4);
          
          viewer.current();

          printf("alpha = %f",alpha);
          printf("center of the previous interpolated point: %f,%f,%f\n", fmat2[3][0],fmat2[3][1],fmat2[3][2]);
          printf("center of the last interpolated point: %f,%f,%f\n", center[0],center[1],center[2]);
          fmat2[3][0] = center[0];
          fmat2[3][1] = center[1];
          fmat2[3][2] = center[2];
          hpld_inter->model(fmat2);
        }
      }
    }
    else
    {
      isHoldOn = false;
      checkPath = false;
      viewer.setPaused(false);
    }
    viewer.current();
    render(&viewer);
    viewer2.current();
    render(&viewer2);
    viewer.current();
    printf("end of an event loop\n");
    printf("viewer: buffer = %d %d, window = %d %d\n", viewer.widthBuffer(), viewer.heightBuffer(), viewer.width(), viewer.height());
    printf("viewer2: buffer = %d %d, window = %d %d\n", viewer2.widthBuffer(), viewer2.heightBuffer(), viewer2.width(), viewer2.height());
  }

  /* clean exit; all okay */
  Hale::done();
 
  airMopOkay(mop);

  return 0;
}
