#include <iostream>
#include <Hale.h>
#include <glm/glm.hpp>

#include "unistd.h" // for sleep()

#include <fstream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "lib/Image.h"

//from cuda_volume_rendering
#define PI 3.14159265


texture<float, 3, cudaReadModeElementType> tex0;  // 3D texture
texture<float, 3, cudaReadModeElementType> tex1;  // 3D texture
cudaArray *d_volumeArray0 = 0;
cudaArray *d_volumeArray1 = 0;

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
__device__
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
__device__
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
__device__
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

__device__
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

__global__
void kernel_cpr(int* dim, int *size, double *center, double *dir1, double *dir2, int nOutChannel, double* imageDouble
        )
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ((i>=size[0]) || (j>=size[1]))
        return;

    int ni = i-size[0]/2;
    int nj = size[1]/2 - j;
    double pointi[3];
    advancePoint(center,dir1,ni,pointi);
    advancePoint(pointi,dir2,nj,pointi);
    double val = tex3DBicubic<float,float>(tex0,pointi[0],pointi[1],pointi[2]);    
    imageDouble[j*size[0]*nOutChannel+i*nOutChannel] = val;
    for (int k=1; k<nOutChannel-1; k++)
      imageDouble[j*size[0]*nOutChannel+i*nOutChannel+k] = 0;
    imageDouble[j*size[0]*nOutChannel+i*nOutChannel+nOutChannel-1] = 1;   
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

void cross(double *u, double *v, double *w)
{
    w[0] = u[1]*v[2]-u[2]*v[1];
    w[1] = u[2]*v[0]-u[0]*v[2];
    w[2] = u[0]*v[1]-u[1]*v[0];
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


float lerp(float y0, float y1, float x0, float x, float x1)
{
  float alpha = (x-x0)/(x1-x0);
  return y0*(1-alpha)+alpha*y1;
}

float linearizeDepth(float depth, float zNear, float zFar)
{
    return (2.0 * zFar * zNear) / (zFar + zNear - depth * (zFar - zNear));
}

float linearizeDepthOrtho(float depth, float zNear, float zFar)
{
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

void render(Hale::Viewer *viewer){
  viewer->draw();
  viewer->bufferSwap();
}

int
main(int argc, const char **argv) {
  const char *me;
  char *err;
  hestOpt *hopt=NULL;
  hestParm *hparm;
  airArray *mop;

  char *name;
  char *texname1, *texname2;
  
  //double dir1[3]={1,0,0};
  //double dir2[3]={0,-1,0};
  double dir1[3],dir2[3];
  //double *dir1,*dir2;

  //tmp fixed track coords, and radius
  double track[3] = {366.653991263,89.6381792864,104.736646409};
  double trackhomo[4];
  trackhomo[0] = track[0];
  trackhomo[1] = track[1];
  trackhomo[2] = track[2];
  trackhomo[3] = 1;
  double trackw[4];
  double radius = 10;

//double *center;
  double center[3];
  //memcpy(center,track,sizeof(double)*3);

  int size[2];
  Nrrd *nin;
  char *outname;

  /* boilerplate hest code */
  me = argv[0];
  mop = airMopNew();
  hparm = hestParmNew();
  airMopAdd(mop, hparm, (airMopper)hestParmFree, airMopAlways);
  /* setting up the command-line options */
  hparm->respFileEnable = AIR_TRUE;
  hparm->noArgsIsNoProblem = AIR_TRUE;

  hestOptAdd(&hopt, "i", "nin", airTypeOther, 1, 1, &nin, "270.nrrd",
             "input volume to render", NULL, NULL, nrrdHestNrrd);

  hestOptAdd(&hopt, "isize", "sx sy", airTypeInt, 2, 2, size, "200 200",
             "output image sizes");

  hestOptAdd(&hopt, "dir1", "x y z", airTypeDouble, 3, 3, dir1, "1 0 0",
             "first direction of the generated image");

  hestOptAdd(&hopt, "dir2", "x y z", airTypeDouble, 3, 3, dir2, "0 -1 0",
             "second direction of the generated image");

  hestOptAdd(&hopt, "center", "x y z", airTypeDouble, 3, 3, center, "366.653991263 89.6381792864 104.736646409",
             "center of the generated image");

  hestOptAdd(&hopt, "o", "name", airTypeString, 1, 1, &outname, "cpr.tga", "name of output image");

  hestParseOrDie(hopt, argc-1, argv+1, hparm,
                 me, "demo program", AIR_TRUE, AIR_TRUE, AIR_TRUE);
  airMopAdd(mop, hopt, (airMopper)hestOptFree, airMopAlways);
  airMopAdd(mop, hopt, (airMopper)hestParseFree, airMopAlways);

  /* Compute threshold (isovalue) */

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
                /* for 2-channel data; this "i" should be "i+1" */
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
                /* for 2-channel data; this "i" should be "i+1" */
                mat_trans[j][i] = nin->axis[i+1].spaceDirection[j];
            }
            mat_trans[i][3] = nin->spaceOrigin[i];
        }
    }
    int channel = 1;
    //int filesize = dim[0]*dim[1]*dim[2]*dim[3]*pixSize;

    float* filemem0 = new float[dim[1]*dim[2]*dim[3]];
    float* filemem1 = new float[dim[1]*dim[2]*dim[3]];

    //filemem = (char*)nin->data;
    for (int i=0; i<dim[1]*dim[2]*dim[3]; i++)
    {
        filemem0[i] = ((short*)nin->data)[i*2];
        filemem1[i] = ((short*)nin->data)[i*2+1];
    }

    double mat_trans_inv[4][4];
    invertMat44(mat_trans,mat_trans_inv);
   //tex3D stuff
    const cudaExtent volumeSize = make_cudaExtent(dim[1], dim[2], dim[3]);

    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMalloc3DArray(&d_volumeArray0, &channelDesc, volumeSize);
    cudaMalloc3DArray(&d_volumeArray1, &channelDesc, volumeSize);

    // --- Copy data to 3D array (host to device)
    cudaMemcpy3DParms copyParams1 = {0};
    copyParams1.srcPtr   = make_cudaPitchedPtr((void*)filemem1, volumeSize.width*pixSize, volumeSize.width, volumeSize.height);
    copyParams1.dstArray = d_volumeArray1;
    copyParams1.extent   = volumeSize;
    copyParams1.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams1);

    cudaMemcpy3DParms copyParams0 = {0};
    copyParams0.srcPtr   = make_cudaPitchedPtr((void*)filemem0, volumeSize.width*pixSize, volumeSize.width, volumeSize.height);
    copyParams0.dstArray = d_volumeArray0;
    copyParams0.extent   = volumeSize;
    copyParams0.kind     = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams0);
    // --- Set texture parameters
    tex1.normalized = false;                      // access with normalized texture coordinates
    tex1.filterMode = cudaFilterModeLinear;      // linear interpolation
    /*
    tex1.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
    tex1.addressMode[1] = cudaAddressModeWrap;
    tex1.addressMode[2] = cudaAddressModeWrap;
    */
    tex1.addressMode[0] = cudaAddressModeBorder;   // wrap texture coordinates
    tex1.addressMode[1] = cudaAddressModeBorder;
    tex1.addressMode[2] = cudaAddressModeBorder;


    tex0.normalized = false;                      // access with normalized texture coordinates
    tex0.filterMode = cudaFilterModeLinear;      // linear interpolation
    /*
    tex0.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
    tex0.addressMode[1] = cudaAddressModeWrap;
    tex0.addressMode[2] = cudaAddressModeWrap;
    */
    tex0.addressMode[0] = cudaAddressModeBorder;   // wrap texture coordinates
    tex0.addressMode[1] = cudaAddressModeBorder;
    tex0.addressMode[2] = cudaAddressModeBorder;
    // --- Bind array to 3D texture
    cudaBindTextureToArray(tex1, d_volumeArray1, channelDesc);
    cudaBindTextureToArray(tex0, d_volumeArray0, channelDesc);
    //-----------

    int nOutChannel = 4;

    double *imageDouble = new double[size[0]*size[1]*nOutChannel];
    //CUDA Var

    int *d_dim;
    cudaMalloc(&d_dim, sizeof(dim));
    cudaMemcpy(d_dim, dim, 4*sizeof(int), cudaMemcpyHostToDevice);

    double *d_dir1;
    cudaMalloc(&d_dir1, sizeof(dir1));
    cudaMemcpy(d_dir1, dir1, 3*sizeof(double), cudaMemcpyHostToDevice);

    double *d_dir2;
    cudaMalloc(&d_dir2, sizeof(dir2));
    cudaMemcpy(d_dir2, dir2, 3*sizeof(double), cudaMemcpyHostToDevice);

    double *d_imageDouble;
    cudaMalloc(&d_imageDouble,sizeof(double)*size[0]*size[1]*nOutChannel);

    int *d_size;
    cudaMalloc(&d_size,2*sizeof(int));
    cudaMemcpy(d_size,size,2*sizeof(int), cudaMemcpyHostToDevice);

    double *d_center;
    cudaMalloc(&d_center,3*sizeof(double));
    cudaMemcpy(d_center,center,3*sizeof(double), cudaMemcpyHostToDevice);


    int numThread1D = 16;
    dim3 threadsPerBlock(numThread1D,numThread1D);
    dim3 numBlocks((size[0]+numThread1D-1)/numThread1D,(size[1]+numThread1D-1)/numThread1D);

    kernel_cpr<<<numBlocks,threadsPerBlock>>>(d_dim, d_size, d_center, d_dir1, d_dir2, nOutChannel, d_imageDouble);

    cudaError_t errCu = cudaGetLastError();
    if (errCu != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(errCu));

    errCu = cudaDeviceSynchronize();
    if (errCu != cudaSuccess) 
        printf("Error Sync: %s\n", cudaGetErrorString(errCu));

    cudaMemcpy(imageDouble, d_imageDouble, sizeof(double)*size[0]*size[1]*nOutChannel, cudaMemcpyDeviceToHost);

    short width = size[0];
    short height = size[1];

    unsigned char *imageQuantized = new unsigned char[size[0]*size[1]*4];
    quantizeImageDouble3D(imageDouble,imageQuantized,4,size[0],size[1]);
    setPlane<unsigned char>(imageQuantized, 4, size[0], size[1], 255, 3);
//end of cuda_rendering

  saveImageWithoutQuantizing<unsigned char>(size[0],size[1],4,imageQuantized,outname);

  airMopOkay(mop);

  return 0;
}
