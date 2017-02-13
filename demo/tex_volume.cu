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
void kernel(int* dim, int *size, double hor_extent, double ver_extent, int channel, int pixSize, double *center, double *viewdir, double *right, double *up, double *light_dir,
        double nc, double fc, double raystep, double refstep, double* mat_trans, double* mat_trans_inv, double* MT_BE_inv, double phongKa, double phongKd, double isoval, double alphamax, double thickness,
        int nOutChannel, double* imageDouble
        )
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ((i>=size[0]) || (j>=size[1]))
        return;

    double hor_ratio = hor_extent/size[0];
    double ver_ratio = ver_extent/size[1];
    int ni = i-size[0]/2;
    int nj = size[1]/2 - j;

    double startPoint1[4];
    startPoint1[3] = 1;
    advancePoint(center,right,ni*ver_ratio,startPoint1);
    double startPoint2[4];
    startPoint2[3] = 1;
    advancePoint(startPoint1,up,nj*hor_ratio,startPoint2);

    memcpy(startPoint1,startPoint2,4*sizeof(double));

    double accColor = 0;
    double transp = 1;    
    double indPoint[4];
    double val;
    double gradi[3];
    double gradw[3];
    double gradw_len;
    //double gradi_len;
    double depth;
    double pointColor;
    double alpha;
    double mipVal = -1;
    double valgfp;

    for (double k=0; k<fc-nc; k+=raystep)
  {
        advancePoint(startPoint1,viewdir,raystep,startPoint2);

        cu_mulMatPoint(mat_trans_inv,startPoint1,indPoint);
        if (cu_isInsideDouble(indPoint[0],indPoint[1],indPoint[2],dim[1],dim[2],dim[3]))
    {

            val = tex3DBicubic<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);
            
            gradi[0] = tex3DBicubic_GX<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);
            gradi[1] = tex3DBicubic_GY<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);
            gradi[2] = tex3DBicubic_GZ<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);

            cu_mulMatPoint3(MT_BE_inv, gradi, gradw);
            gradw_len = lenVec(gradw,3);

            //negating and normalizing
            for (int l=0; l<3; l++)
                gradw[l] = -gradw[l]/gradw_len;

            depth = (k*1.0+1)/(fc*1.0-nc);

            pointColor = phongKa + depth*phongKd*max(0.0f,dotProduct(gradw,light_dir,3));
            alpha = cu_computeAlpha(val, gradw_len, isoval, alphamax, thickness);
            alpha = 1 - pow(1-alpha,raystep/refstep);
            transp *= (1-alpha);
            accColor = accColor*(1-alpha) + pointColor*alpha;

            valgfp = tex3DBicubic<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);

            mipVal = max(mipVal,valgfp*cu_inAlpha(val,gradw_len,isoval,thickness));
    }

        memcpy(startPoint1,startPoint2,4*sizeof(double));
  }
    
    double accAlpha = 1 - transp;
    
    if (accAlpha>0)
    {        
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel] = accColor/accAlpha;
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel+1] = mipVal;
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel+2] = 0;
    }
    else
    {        
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel] = accColor;
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel+1] = mipVal;
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel+2] = 0;        
    }
    imageDouble[j*size[0]*nOutChannel+i*nOutChannel+nOutChannel-1] = accAlpha;    
}

__global__
void kernel_peak(int* dim, int *size, double hor_extent, double ver_extent, int channel, int pixSize, double *center, double *viewdir, double *right, double *up, double *light_dir,
        double nc, double fc, double raystep, double refstep, double* mat_trans, double* mat_trans_inv, double* MT_BE_inv, double* M_BE_inv, double phongKa, double phongKd, double isoval, double alphamax, double thickness,
        int nOutChannel, double* imageDouble
        )
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ((i>=size[0]) || (j>=size[1]))
        return;

    double hor_ratio = hor_extent/size[0];
    double ver_ratio = ver_extent/size[1];
    int ni = i-size[0]/2;
    int nj = size[1]/2 - j;

    double startPoint1[4];
    startPoint1[3] = 1;
    advancePoint(center,right,ni*ver_ratio,startPoint1);
    double startPoint2[4];
    startPoint2[3] = 1;
    advancePoint(startPoint1,up,nj*hor_ratio,startPoint2);

    memcpy(startPoint1,startPoint2,4*sizeof(double));

    double accColor = 0;
    double transp = 1;    
    double indPoint[4];
    double val;
    double gradi[3];
    double gradw[3];
    double gradgfpi[3];
    double gradgfpw[3];
    double gradw_len;
    //double gradi_len;
    double depth;
    double pointColor;
    double alpha;
    double mipVal = -1;
    double valgfp;
    double hessian[9];
    double hessian_w[9];
    double hessian_w33[3][3];
    double hessian_w33inv[3][3];
    double hessian_winv[9];
    double peakdis[3];
    double len_peakdis;
    double pointColorGFP;
    double alphaGFP;
    double transpGFP = 1;
    double accColorGFP = 0;

    for (double k=0; k<fc-nc; k+=raystep)
  {
        advancePoint(startPoint1,viewdir,raystep,startPoint2);

        cu_mulMatPoint(mat_trans_inv,startPoint1,indPoint);
        if (cu_isInsideDouble(indPoint[0],indPoint[1],indPoint[2],dim[1],dim[2],dim[3]))
    {

            val = tex3DBicubic<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);
            
            gradi[0] = tex3DBicubic_GX<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);
            gradi[1] = tex3DBicubic_GY<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);
            gradi[2] = tex3DBicubic_GZ<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);

            cu_mulMatPoint3(MT_BE_inv, gradi, gradw);
            gradw_len = lenVec(gradw,3);

            //negating and normalizing
            for (int l=0; l<3; l++)
                gradw[l] = -gradw[l]/gradw_len;

            depth = (k*1.0+1)/(fc*1.0-nc);

            pointColor = phongKa + depth*phongKd*max(0.0f,dotProduct(gradw,light_dir,3));
            alpha = cu_computeAlpha(val, gradw_len, isoval, alphamax, thickness);
            alpha = 1 - pow(1-alpha,raystep/refstep);
            transp *= (1-alpha);
            accColor = accColor*(1-alpha) + pointColor*alpha;

            valgfp = tex3DBicubic<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);

            mipVal = max(mipVal,valgfp*cu_inAlpha(val,gradw_len,isoval,thickness));

            if (alpha>0)
            {
              /*
                hessian[cu_getIndex2(0,0,3,3)]=tex3DBicubic_GGX<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);
                hessian[cu_getIndex2(0,1,3,3)]=tex3DBicubic_GYGX<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);
                hessian[cu_getIndex2(0,2,3,3)]=tex3DBicubic_GZGX<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);
                hessian[cu_getIndex2(1,1,3,3)]=tex3DBicubic_GGY<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);
                hessian[cu_getIndex2(1,2,3,3)]=tex3DBicubic_GZGY<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);
                hessian[cu_getIndex2(2,2,3,3)]=tex3DBicubic_GGZ<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);

                hessian[cu_getIndex2(1,0,3,3)] = hessian[cu_getIndex2(0,1,3,3)];
                hessian[cu_getIndex2(2,0,3,3)] = hessian[cu_getIndex2(0,2,3,3)];
                hessian[cu_getIndex2(2,1,3,3)] = hessian[cu_getIndex2(1,2,3,3)];
                */
                computeHessian(hessian,indPoint);

                /*
                double mattmp[9];
                memset(mattmp,0,9*sizeof(double));
                mulMat3(hessian,M_BE_inv,mattmp);
                mulMat3(MT_BE_inv,mattmp,hessian_w);

                memcpy(hessian_w33,hessian_w,sizeof(double)*9);
                invertMat33(hessian_w33,hessian_w33inv);
                memcpy(hessian_winv,hessian_w33inv,sizeof(double)*9);
                */

                gradgfpi[0] = tex3DBicubic_GX<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);
                gradgfpi[1] = tex3DBicubic_GY<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);
                gradgfpi[2] = tex3DBicubic_GZ<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);
                cu_mulMatPoint3(MT_BE_inv, gradgfpi, gradgfpw);
                
                double hessian_inv[9];
                memcpy(hessian_w33,hessian,sizeof(double)*9);
                invertMat33(hessian_w33,hessian_w33inv);
                memcpy(hessian_inv,hessian_w33inv,sizeof(double)*9);
                
                //cu_mulMatPoint3(hessian_winv,gradgfpw,peakdis);
                cu_mulMatPoint3(hessian_inv,gradgfpi,peakdis);
                scaleVector(peakdis,3,-1);
                len_peakdis = lenVec(peakdis,3);
                double critpoint[3];
                addVector(indPoint,peakdis,critpoint,3);

                //see if it is maximum point
                computeHessian(hessian,critpoint);
                double eigenval[3];
                eigenOfHess(hessian,eigenval);
                if (eigenval[0]<0 && eigenval[1]<0 && eigenval[2]<0)
                {                
                  pointColorGFP = phongKa + depth*phongKd*max(0.0f,dotProduct(gradgfpw,light_dir,3));
                  alphaGFP = cu_inAlphaX(len_peakdis-19,thickness);//cu_computeAlpha(val, gradw_len, isoval, alphamax, thickness);
                  alphaGFP = 1 - pow(1-alphaGFP,raystep/refstep);
                  transpGFP *= (1-alphaGFP);
                  accColorGFP = accColorGFP*(1-alphaGFP) + pointColorGFP*alphaGFP;
                }
            }
    }

        memcpy(startPoint1,startPoint2,4*sizeof(double));
  }
    
    double accAlpha = 1 - transp;
    double accAlphaGFP = 1 - transpGFP;
    
    if (accAlpha>0)
    {        
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel] = accColor/accAlpha;
    }
    else
    {
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel] = accColor;
    }
    if (accAlphaGFP>0)
    {
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel+1] = accColorGFP/accAlphaGFP;
    }
    else
    {
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel+1] = accColorGFP;  
    }
    imageDouble[j*size[0]*nOutChannel+i*nOutChannel+2] = 0;
        
    imageDouble[j*size[0]*nOutChannel+i*nOutChannel+nOutChannel-1] = accAlpha;    
}


__global__
void kernel_combined(int* dim, int *size, double hor_extent, double ver_extent, int channel, int pixSize, double *center, double *viewdir, double *right, double *up, double *light_dir,
        double nc, double fc, double raystep, double refstep, double* mat_trans, double* mat_trans_inv, double* MT_BE_inv, double phongKa, double phongKd, double isoval, double alphamax, double thickness,
        double trackx, double tracky, double trackz, double radius, int nOutChannel, double* imageDouble, int* imageMask
        )
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int j = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ((i>=size[0]) || (j>=size[1]))
        return;

    double hor_ratio = hor_extent/size[0];
    double ver_ratio = ver_extent/size[1];
    int ni = i-size[0]/2;
    int nj = size[1]/2 - j;

    double startPoint1[4];
    startPoint1[3] = 1;
    advancePoint(center,right,ni*ver_ratio,startPoint1);
    double startPoint2[4];
    startPoint2[3] = 1;
    advancePoint(startPoint1,up,nj*hor_ratio,startPoint2);

    memcpy(startPoint1,startPoint2,4*sizeof(double));

    double accColor = 0;
    double transp = 1;    
    double indPoint[4];
    double val;
    double gradi[3];
    double gradw[3];
    double gradw_len;
    //double gradi_len;
    double depth;
    double pointColor;
    double alpha;
    double mipVal = -1;
    double valgfp;
    double mipValR = -1;
    double mipValG = -1;

    cu_mulMatPoint(mat_trans_inv,startPoint1,indPoint);
    double vecview[4];
    memcpy(vecview,viewdir,3*sizeof(double));
    vecview[3] = 0;
    double vecviewi[4];
    cu_mulMatPoint(mat_trans_inv,vecview,vecviewi);
    double vecdiff[3];
    vecdiff[0] = indPoint[0]-trackx;
    vecdiff[1] = indPoint[1]-tracky;
    vecdiff[2] = indPoint[2]-trackz;
    double lendiff = lenVec(vecdiff,3);
    double lenview = lenVec(vecviewi,3);
    double dotres = dotProduct(vecdiff,vecviewi,3);
    double diss = (lendiff*lendiff*lenview*lenview-dotres*dotres)/(lenview*lenview);

    bool isCloseTrack = (diss<=(radius*radius));
    imageMask[j*size[0]+i] =(int)isCloseTrack;

    for (double k=0; k<fc-nc; k+=raystep)
    {
        advancePoint(startPoint1,viewdir,raystep,startPoint2);

        cu_mulMatPoint(mat_trans_inv,startPoint1,indPoint);
        if (cu_isInsideDouble(indPoint[0],indPoint[1],indPoint[2],dim[1],dim[2],dim[3]))
        {
            if (!isCloseTrack)
            {
                val = tex3DBicubic<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);
                
                gradi[0] = tex3DBicubic_GX<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);
                gradi[1] = tex3DBicubic_GY<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);
                gradi[2] = tex3DBicubic_GZ<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);

                cu_mulMatPoint3(MT_BE_inv, gradi, gradw);
                gradw_len = lenVec(gradw,3);

                //negating and normalizing
                for (int l=0; l<3; l++)
                    gradw[l] = -gradw[l]/gradw_len;

                depth = (k*1.0+1)/(fc*1.0-nc);

                pointColor = phongKa + depth*phongKd*max(0.0f,dotProduct(gradw,light_dir,3));
                alpha = cu_computeAlpha(val, gradw_len, isoval, alphamax, thickness);
                alpha = 1 - pow(1-alpha,raystep/refstep);
                transp *= (1-alpha);
                accColor = accColor*(1-alpha) + pointColor*alpha;

                valgfp = tex3DBicubic<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);

                mipVal = max(mipVal,valgfp*cu_inAlpha(val,gradw_len,isoval,thickness));
            }
            else
            {
                double diss = diss2P(indPoint[0],indPoint[1],indPoint[2],trackx,tracky,trackz);
                if (diss<=(radius*radius))
                {
                    if (((radius*radius)-diss)<1)
                    {
                        gradi[0] = indPoint[0]-trackx;
                        gradi[1] = indPoint[1]-tracky;
                        gradi[2] = indPoint[2]-trackz;

                        cu_mulMatPoint3(MT_BE_inv, gradi, gradw);
                            gradw_len = lenVec(gradw,3);

                        //negating and normalizing
                        for (int l=0; l<3; l++)
                            gradw[l] = -gradw[l]/gradw_len;

                        pointColor = phongKa + depth*phongKd*max(0.0f,dotProduct(gradw,light_dir,3));
                    }
                    val = tex3DBicubic<float,float>(tex1,indPoint[0],indPoint[1],indPoint[2]);
                    mipValR = max(mipValR,val);
                    valgfp = tex3DBicubic<float,float>(tex0,indPoint[0],indPoint[1],indPoint[2]);
                    mipValG = max(mipValG,valgfp);
                }
            }
        }

        memcpy(startPoint1,startPoint2,4*sizeof(double));
    }
    
    if (!isCloseTrack)
    {
        double accAlpha = 1 - transp;
        
        if (accAlpha>0)
        {        
            imageDouble[j*size[0]*nOutChannel+i*nOutChannel] = accColor/accAlpha;
            imageDouble[j*size[0]*nOutChannel+i*nOutChannel+1] = mipVal;
            imageDouble[j*size[0]*nOutChannel+i*nOutChannel+2] = 0;
        }
        else
        {        
            imageDouble[j*size[0]*nOutChannel+i*nOutChannel] = accColor;
            imageDouble[j*size[0]*nOutChannel+i*nOutChannel+1] = mipVal;
            imageDouble[j*size[0]*nOutChannel+i*nOutChannel+2] = 0;        
        }
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel+nOutChannel-1] = accAlpha;    
    }
    else
    {
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel] = 0;
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel+1] = mipValG;
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel+2] = pointColor;        
        imageDouble[j*size[0]*nOutChannel+i*nOutChannel+nOutChannel-1] = 255;    
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

  //tmp fixed track coords, and radius
  double track[3] = {366.653991263,89.6381792864,104.736646409};
  double trackhomo[4];
  trackhomo[0] = track[0];
  trackhomo[1] = track[1];
  trackhomo[2] = track[2];
  trackhomo[3] = 1;
  double trackw[4];
  double radius = 10;

  /* variables learned via hest */
  //float camfr[3], camat[3], camup[3], camnc, camfc, camFOV;
  double fr[3], at[3], up[3], nc, fc, fov, light_dir[3], isoval, raystep, refstep, thickness, alphamax, phongKa, phongKd;
  int size[2];
  int camortho = 0;
  Nrrd *nin;
  //unsigned int camsize[2];
  //double isovalue, sliso;

  //double evec[9], eval[3];
  //double mean[3], cov[9];

  //Nrrd *nout = nrrdNew();
  //unsigned int bins = 2000;
  //int type;

  Hale::debugging = 1;

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
    hestOptAdd(&hopt, "fr", "from", airTypeDouble, 3, 3, fr, "-50 0 0",
               "look-from point");
    hestOptAdd(&hopt, "at", "at", airTypeDouble, 3, 3, at, "0 0 0",
               "look-at point");
    hestOptAdd(&hopt, "up", "up", airTypeDouble, 3, 3, up, "0 0 1",
               "pseudo-up vector");
    hestOptAdd(&hopt, "nc", "near-clip", airTypeDouble, 1, 1, &nc, "-50",
               "near clipping plane");
    hestOptAdd(&hopt, "fc", "far-clip", airTypeDouble, 1, 1, &fc, "50",
               "far clipping plane");
    hestOptAdd(&hopt, "fov", "FOV", airTypeDouble, 1, 1, &fov, "10",
               "field-of-view");
    hestOptAdd(&hopt, "ldir", "direction", airTypeDouble, 3, 3, light_dir, "-1 0 0",
               "direction towards light");
    hestOptAdd(&hopt, "isize", "sx sy", airTypeInt, 2, 2, size, "200 200",
               "output image sizes");
    hestOptAdd(&hopt, "iso", "iso-value", airTypeDouble, 1, 1, &isoval, "0",
               "iso-value");
    hestOptAdd(&hopt, "step", "ray-step", airTypeDouble, 1, 1, &raystep, "0.1",
               "ray traversing step");
    hestOptAdd(&hopt, "refstep", "ref-step", airTypeDouble, 1, 1, &refstep, "1",
               "ref-step");
    hestOptAdd(&hopt, "thick", "thickness", airTypeDouble, 1, 1, &thickness, "0.5",
               "thickness around iso-value");
    hestOptAdd(&hopt, "alpha", "max-alpha", airTypeDouble, 1, 1, &alphamax, "1",
               "maximum value of alpha");
    hestOptAdd(&hopt, "phongKa", "phong-Ka", airTypeDouble, 1, 1, &phongKa, "0.2",
               "Ka value of Phong shading");
    hestOptAdd(&hopt, "phongKd", "phong-Kd", airTypeDouble, 1, 1, &phongKd, "0.8",
               "Kd value of Phong shading");

  hestOptAdd(&hopt, "tex1", "texname1", airTypeString, 1, 1, &texname1, "tex1.png", "name of first texture image");
  hestOptAdd(&hopt, "tex2", "texname2", airTypeString, 1, 1, &texname2, "tex2.png", "name of second texture image");
  camortho = 1;

  hestParseOrDie(hopt, argc-1, argv+1, hparm,
                 me, "demo program", AIR_TRUE, AIR_TRUE, AIR_TRUE);
  airMopAdd(mop, hopt, (airMopper)hestOptFree, airMopAlways);
  airMopAdd(mop, hopt, (airMopper)hestParseFree, airMopAlways);

  /* Compute threshold (isovalue) */

  /* then create empty scene */
  Hale::init();
  Hale::Scene scene;
  /* then create viewer (in order to create the OpenGL context) */
  Hale::Viewer viewer(size[0], size[1], "Volume_Rendering_Hale", &scene);
  //viewer.lightDir(glm::vec3(-1.0f, 1.0f, 3.0f));
  viewer.lightDir(glm::vec3(light_dir[0], light_dir[1], light_dir[2]));
  viewer.camera.init(glm::vec3(fr[0], fr[1], fr[2]),
                     glm::vec3(at[0], at[1], at[2]),
                     glm::vec3(up[0], up[1], up[2]),
                     fov, (float)size[0]/size[1],
                     nc, fc, camortho);
  viewer.refreshCB((Hale::ViewerRefresher)render);
  viewer.refreshData(&viewer);
  viewer.current();

  Hale::Program *newprog = new Hale::Program("texdemo-vert.glsl","texdemo-frag.glsl");
  newprog->compile();
  newprog->bindAttribute(Hale::vertAttrIdxXYZW, "positionVA");
  newprog->bindAttribute(Hale::vertAttrIdxRGBA, "colorVA");
  newprog->bindAttribute(Hale::vertAttrIdxNorm, "normalVA");
  newprog->bindAttribute(Hale::vertAttrIdxTex2, "tex2VA");
  newprog->link();  

  //--cuda_rendering
     //process input
    normalize(light_dir,3);

    unsigned int pixSize = 1;
    cudaChannelFormatDesc channelDesc;
    /*
    switch (nin->type)
    {
        case nrrdTypeFloat:
            pixSize = sizeof(float);
            channelDesc = cudaCreateChannelDesc<float>();
            break;
        case nrrdTypeShort:
            pixSize = sizeof(short);
            channelDesc = cudaCreateChannelDesc<short>();
            break;
        case nrrdTypeDouble:
            pixSize = sizeof(double);
            channelDesc = cudaCreateChannelDesc<double>();
            break;            
        case nrrdTypeInt:
            pixSize = sizeof(int);
            channelDesc = cudaCreateChannelDesc<int>();
            break;            
        default:
            break;
    }
    */
    pixSize = sizeof(float);
    channelDesc = cudaCreateChannelDesc<float>();
    /* 2-channel data will have:
       4 == nin->dim
       2 == nin->axis[0].size
       3 == nin->spaceDim */
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

    double vb0[4] = {0,0,0,1};
    double vb1[4] = {1,0,0,1};
    double vb2[4] = {0,1,0,1};
    double vb3[4] = {0,0,1,1};
    double ve0[4],ve1[4],ve2[4],ve3[4];
    mulMatPoint(mat_trans,vb0,ve0);
    mulMatPoint(mat_trans,vb1,ve1);
    mulMatPoint(mat_trans,vb2,ve2);
    mulMatPoint(mat_trans,vb3,ve3);
    subtractVec(ve1,ve0,ve1,3);
    subtractVec(ve2,ve0,ve2,3);
    subtractVec(ve3,ve0,ve3,3);

    double MT_BE[3][3];
    MT_BE[0][0] = dotProduct(vb1,ve1,3);
    MT_BE[0][1] = dotProduct(vb2,ve1,3);
    MT_BE[0][2] = dotProduct(vb3,ve1,3);
    MT_BE[1][0] = dotProduct(vb1,ve2,3);
    MT_BE[1][1] = dotProduct(vb2,ve2,3);
    MT_BE[1][2] = dotProduct(vb3,ve2,3);
    MT_BE[2][0] = dotProduct(vb1,ve3,3);
    MT_BE[2][1] = dotProduct(vb2,ve3,3);
    MT_BE[2][2] = dotProduct(vb3,ve3,3);

    double MT_BE_inv[3][3];
    invertMat33(MT_BE,MT_BE_inv);
    double M_BE_inv[3][3];
    transposeMat33(MT_BE_inv,M_BE_inv);

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
    tex1.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
    tex1.addressMode[1] = cudaAddressModeWrap;
    tex1.addressMode[2] = cudaAddressModeWrap;

    tex0.normalized = false;                      // access with normalized texture coordinates
    tex0.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex0.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
    tex0.addressMode[1] = cudaAddressModeWrap;
    tex0.addressMode[2] = cudaAddressModeWrap;
    // --- Bind array to 3D texture
    cudaBindTextureToArray(tex1, d_volumeArray1, channelDesc);
    cudaBindTextureToArray(tex0, d_volumeArray0, channelDesc);
    //-----------

    normalize(up,3);

    double viewdir[3];
    subtractVec(at,fr,viewdir,3);
    double viewdis = lenVec(viewdir,3);
    double ver_extent = 2*viewdis*tan((fov/2)*PI/180);
    double hor_extent = (ver_extent/size[1])*size[0];
    normalize(viewdir,3);

    double nviewdir[3];
    memcpy(nviewdir,viewdir,sizeof(viewdir));
    negateVec(nviewdir,3);

    double right[3];
    cross(up,nviewdir,right);
    normalize(right,3);

    //correcting the up vector
    cross(nviewdir,right,up);
    normalize(up,3);

    double center[3];
    advancePoint(at,viewdir,nc,center);

    int nOutChannel = 4;

    double *imageDouble = new double[size[0]*size[1]*nOutChannel];
    int *imageMask = new int[size[0]*size[1]];

    //CUDA Var

    int *d_dim;
    cudaMalloc(&d_dim, sizeof(dim));
    cudaMemcpy(d_dim, dim, 4*sizeof(int), cudaMemcpyHostToDevice);

    double *d_imageDouble;
    cudaMalloc(&d_imageDouble,sizeof(double)*size[0]*size[1]*nOutChannel);

    int *d_imageMask;
    cudaMalloc(&d_imageMask,sizeof(int)*size[0]*size[1]);

    int *d_size;
    cudaMalloc(&d_size,2*sizeof(int));
    cudaMemcpy(d_size,size,2*sizeof(int), cudaMemcpyHostToDevice);

    double *d_center;
    cudaMalloc(&d_center,3*sizeof(double));
    cudaMemcpy(d_center,center,3*sizeof(double), cudaMemcpyHostToDevice);

    double *d_viewdir;
    cudaMalloc(&d_viewdir,3*sizeof(double));
    cudaMemcpy(d_viewdir,viewdir,3*sizeof(double), cudaMemcpyHostToDevice);

    double *d_up;
    cudaMalloc(&d_up,3*sizeof(double));
    cudaMemcpy(d_up,up,3*sizeof(double), cudaMemcpyHostToDevice);

    double *d_right;
    cudaMalloc(&d_right,3*sizeof(double));
    cudaMemcpy(d_right,right,3*sizeof(double), cudaMemcpyHostToDevice);

    double *d_light_dir;
    cudaMalloc(&d_light_dir,3*sizeof(double));
    cudaMemcpy(d_light_dir,light_dir,3*sizeof(double), cudaMemcpyHostToDevice);

    double* d_mat_trans;
    cudaMalloc(&d_mat_trans,16*sizeof(double));
    cudaMemcpy(d_mat_trans,&mat_trans[0][0],16*sizeof(double), cudaMemcpyHostToDevice);

    double* d_MT_BE_inv;
    cudaMalloc(&d_MT_BE_inv,9*sizeof(double));
    cudaMemcpy(d_MT_BE_inv,&MT_BE_inv[0][0],9*sizeof(double), cudaMemcpyHostToDevice);

    double* d_M_BE_inv;
    cudaMalloc(&d_M_BE_inv,9*sizeof(double));
    cudaMemcpy(d_M_BE_inv,&M_BE_inv[0][0],9*sizeof(double), cudaMemcpyHostToDevice);

    double* d_mat_trans_inv;
    cudaMalloc(&d_mat_trans_inv,16*sizeof(double));
    cudaMemcpy(d_mat_trans_inv,&mat_trans_inv[0][0],16*sizeof(double), cudaMemcpyHostToDevice);

    int numThread1D = 16;
    dim3 threadsPerBlock(numThread1D,numThread1D);
    dim3 numBlocks((size[0]+numThread1D-1)/numThread1D,(size[1]+numThread1D-1)/numThread1D);
/*
    kernel<<<numBlocks,threadsPerBlock>>>(d_dim, d_size, hor_extent, ver_extent, channel, pixSize,
                                          d_center, d_viewdir, d_right, d_up, d_light_dir, nc, fc, raystep, refstep, d_mat_trans,
                                          d_mat_trans_inv, d_MT_BE_inv, phongKa, phongKd, isoval, alphamax, thickness, nOutChannel, d_imageDouble                                          
                                          );
 */

    kernel_peak<<<numBlocks,threadsPerBlock>>>(d_dim, d_size, hor_extent, ver_extent, channel, pixSize,
                                          d_center, d_viewdir, d_right, d_up, d_light_dir, nc, fc, raystep, refstep, d_mat_trans,
                                          d_mat_trans_inv, d_MT_BE_inv, d_M_BE_inv, phongKa, phongKd, isoval, alphamax, thickness, nOutChannel, d_imageDouble                                          
                                          );

/*
    kernel_combined<<<numBlocks,threadsPerBlock>>>(d_dim, d_size, hor_extent, ver_extent, channel, pixSize,
                                          d_center, d_viewdir, d_right, d_up, d_light_dir, nc, fc, raystep, refstep, d_mat_trans,
                                          d_mat_trans_inv, d_MT_BE_inv, phongKa, phongKd, isoval, alphamax, thickness, 
                                          track[0],track[1],track[2],radius,nOutChannel, d_imageDouble, d_imageMask                        
                                          );
*/

    cudaError_t errCu = cudaGetLastError();
    if (errCu != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(errCu));

    errCu = cudaDeviceSynchronize();
    if (errCu != cudaSuccess) 
        printf("Error Sync: %s\n", cudaGetErrorString(errCu));

    cudaMemcpy(imageDouble, d_imageDouble, sizeof(double)*size[0]*size[1]*nOutChannel, cudaMemcpyDeviceToHost);
    cudaMemcpy(imageMask, d_imageMask, sizeof(int)*size[0]*size[1], cudaMemcpyDeviceToHost);

    short width = size[0];
    short height = size[1];

    //double *imageSave = new double[size[0]*size[1]];
    unsigned char *imageQuantized = new unsigned char[size[0]*size[1]*4];
    unsigned char *imageQuantizedMask = new unsigned char[size[0]*size[1]*4];
    unsigned char *imageQuantizedNoB = new unsigned char[size[0]*size[1]*4];
    unsigned char *imageQuantizedMaskNoB = new unsigned char[size[0]*size[1]*4];
    unsigned char *imageQuantizedGreen = new unsigned char[size[0]*size[1]*4];
    quantizeImageDouble3D(imageDouble,imageQuantized,4,size[0],size[1]);
    applyMask(imageQuantized,4,size[0],size[1],imageMask,imageQuantizedMask);
    removeChannel(imageQuantized,4,size[0],size[1],2,imageQuantizedNoB);
    removeChannel(imageQuantizedMask,4,size[0],size[1],2,imageQuantizedMaskNoB);
    removeChannel(imageQuantizedNoB,4,size[0],size[1],0,imageQuantizedGreen);
    setPlane<unsigned char>(imageQuantizedGreen, 4, size[0], size[1], 255, 3);
//end of cuda_rendering

  limnPolyData *lpld = limnPolyDataNew();

  limnPolyDataSquare(lpld, 1 << limnPolyDataInfoNorm | 1 << limnPolyDataInfoTex2);
  
  Hale::Polydata hpld(lpld, true,
                        NULL,
                       "square");
  hpld.program(newprog);
  //hpld.setTexture((char*)"myTextureSampler",nimg);
  //hpld.setTexture((char*)"myTextureSampler",(unsigned char *)nimg->data,300,300,3);
  hpld.setTexture((char*)"myTextureSampler",(unsigned char *)imageQuantized,size[0],size[1],4);
  scene.add(&hpld);

  saveImage<unsigned char>(size[0],size[1],4,imageQuantized,"img.tga");
  saveImageWithoutQuantizing<unsigned char>(size[0],size[1],4,imageQuantizedGreen,"green.tga");
/*
  limnPolyData *lpld2 = limnPolyDataNew();
  limnPolyDataSquare(lpld2, 1 << limnPolyDataInfoNorm | 1 << limnPolyDataInfoTex2);
  Hale::Polydata hpld2(lpld2, true,
                        NULL,
                       "square");

  hpld2.program(newprog);  
  //hpld2.setTexture((char*)"myTextureSampler",nimg2);
  hpld2.setTexture((char*)"myTextureSampler",(unsigned char *)nimg2->data,300,300,3);
  glm::mat4 tmat = glm::mat4();
  tmat[3][2] = 1;
  hpld2.model(tmat);
  scene.add(&hpld2);
*/
  scene.drawInit();
  render(&viewer);


  //--------------------testing another scene with simple square--------
  /*
  float camfr[3] = {0,0,8}, camat[3] = {0,0,0}, camup[3] = {1,0,0}, camnc=-1, camfc=1, camFOV=20;
  camortho = 1;
  unsigned int camsize[2] = {640,480};
  Hale::Scene scene2;
  Hale::Viewer viewer2(camsize[0], camsize[1], "Iso", &scene2);
  viewer2.lightDir(glm::vec3(-1.0f, 1.0f, 3.0f));
  viewer2.camera.init(glm::vec3(camfr[0], camfr[1], camfr[2]),
                     glm::vec3(camat[0], camat[1], camat[2]),
                     glm::vec3(camup[0], camup[1], camup[2]),
                     camFOV, (float)camsize[0]/camsize[1],
                     camnc, camfc, camortho);
  viewer2.refreshCB((Hale::ViewerRefresher)render);
  viewer2.refreshData(&viewer2);
  viewer2.current();
*/
  
  /*
  limnPolyData *lpld2 = limnPolyDataNew();
  limnPolyDataSquare(lpld2, 1 << limnPolyDataInfoNorm);

  Hale::Polydata hpld2(lpld2, true,
                       Hale::ProgramLib(Hale::preprogramAmbDiffSolid),
                       "square");
  hpld2.colorSolid(1,0.5,0.5);
  mulMatPoint(mat_trans,trackhomo,trackw);
  glm::mat4 mmat = glm::mat4();
  mmat[0][0] = 100;
  mmat[1][1] = 100;
  mmat[3][0] = trackw[0];
  mmat[3][1] = trackw[1];
  mmat[3][2] = trackw[2];
  printf("Tracked Point world-space coordinates: %f %f %f\n",trackw[0],trackw[1],trackw[2]);
  hpld2.model(mmat);
  scene.remove(&hpld);
  scene.add(&hpld2);
  scene.drawInit();
  glDepthMask(GL_TRUE);
  glDepthFunc(GL_ALWAYS);
  glDepthRange(0.0f, 1.0f);
  render(&viewer);
  //viewer.draw();
  */
  
  //----------------------------
  
/*
  GLfloat* zbuffer = new GLfloat[size[0]*size[1]];
  glReadPixels(0,0,size[0],size[1],GL_DEPTH_COMPONENT,GL_FLOAT,zbuffer);  
  printf("Z-buffer\n");

  float minz=1000,maxz=-1000;
  for (int i=0; i<size[0]*size[1]; i++)
  {
    zbuffer[i] = linearizeDepthOrtho(lerp(-1,1,0,zbuffer[i],1),nc,fc);
    if (zbuffer[i]<minz)
        minz = zbuffer[i];
    if (zbuffer[i]>maxz)
        maxz = zbuffer[i];
  }
  printf("minmaxz = (%f,%f)\n",minz,maxz );
  saveImage<GLfloat>(size[0],size[1],1,zbuffer,"depth.tga");
    //printf("%f ", zbuffer[i]);
  //viewer.bufferSwap();

  return 0;

  //int count=0;
  */

  glm::vec3 preFrom = viewer.camera.from();
  glm::vec3 preAt = viewer.camera.at();
  glm::vec3 preUp = viewer.camera.up();
  bool isMasked = false;
  bool stateBKey = false;

  while(!Hale::finishing){
    glfwWaitEvents();
  //hpld.replaceLastTexture((unsigned char *)nimg2->data,200,200,3);    
    if ((viewer.camera.from() != preFrom || viewer.camera.at()!=preAt || viewer.camera.up()!=preUp)
        && (viewer.isMouseReleased()))
    {
        printf("------------------------------\n");
        printf("from = (%f,%f,%f)\n", preFrom.x,preFrom.y,preFrom.z);
        printf("at = (%f,%f,%f)\n", preAt.x,preAt.y,preAt.z);
        printf("up = (%f,%f,%f)\n", preUp.x,preUp.y,preUp.z);

        preFrom = viewer.camera.from();
        preAt = viewer.camera.at();
        preUp = viewer.camera.up();

        printf("----------After changing---------------\n");
        printf("from = (%f,%f,%f)\n", preFrom.x,preFrom.y,preFrom.z);
        printf("at = (%f,%f,%f)\n", preAt.x,preAt.y,preAt.z);
        printf("up = (%f,%f,%f)\n", preUp.x,preUp.y,preUp.z);        
        /*
        if (count)
            hpld.replaceLastTexture((unsigned char *)nimg2->data,300,300,3);
        else
            hpld.replaceLastTexture((unsigned char *)nimg->data,300,300,3);
        count = 1-count;
        */
        up[0] = preUp.x;
        up[1] = preUp.y;
        up[2] = preUp.z;
        at[0] = preAt.x;
        at[1] = preAt.y;
        at[2] = preAt.z;
        fr[0] = preFrom.x;
        fr[1] = preFrom.y;
        fr[2] = preFrom.z;
        normalize(up,3);

        subtractVec(at,fr,viewdir,3);
        viewdis = lenVec(viewdir,3);
        ver_extent = 2*viewdis*tan((fov/2)*PI/180);
        hor_extent = (ver_extent/size[1])*size[0];
        normalize(viewdir,3);

        memcpy(nviewdir,viewdir,sizeof(viewdir));
        negateVec(nviewdir,3);

        cross(up,nviewdir,right);
        normalize(right,3);

        //correcting the up vector
        cross(nviewdir,right,up);
        normalize(up,3);

        advancePoint(at,viewdir,nc,center);

        //CUDA Var      
        cudaMemcpy(d_center,center,3*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_viewdir,viewdir,3*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_up,up,3*sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_right,right,3*sizeof(double), cudaMemcpyHostToDevice);

/*
        kernel<<<numBlocks,threadsPerBlock>>>(d_dim, d_size, hor_extent, ver_extent, channel, pixSize,
                                              d_center, d_viewdir, d_right, d_up, d_light_dir, nc, fc, raystep, refstep, d_mat_trans,
                                              d_mat_trans_inv, d_MT_BE_inv, phongKa, phongKd, isoval, alphamax, thickness, nOutChannel, d_imageDouble                                          
                                              );
*/
        kernel_combined<<<numBlocks,threadsPerBlock>>>(d_dim, d_size, hor_extent, ver_extent, channel, pixSize,
                                              d_center, d_viewdir, d_right, d_up, d_light_dir, nc, fc, raystep, refstep, d_mat_trans,
                                              d_mat_trans_inv, d_MT_BE_inv, phongKa, phongKd, isoval, alphamax, thickness, 
                                              track[0],track[1],track[2],radius,nOutChannel, d_imageDouble, d_imageMask                        
                                              );
        
        errCu = cudaGetLastError();
        if (errCu != cudaSuccess) 
            printf("Error: %s\n", cudaGetErrorString(errCu));

        errCu = cudaDeviceSynchronize();
        if (errCu != cudaSuccess) 
            printf("Error Sync: %s\n", cudaGetErrorString(errCu));

        cudaMemcpy(imageDouble, d_imageDouble, sizeof(double)*size[0]*size[1]*nOutChannel, cudaMemcpyDeviceToHost);
        cudaMemcpy(imageMask, d_imageMask, sizeof(int)*size[0]*size[1], cudaMemcpyDeviceToHost);
        quantizeImageDouble3D(imageDouble,imageQuantized,4,size[0],size[1]);
        applyMask(imageQuantized,4,size[0],size[1],imageMask,imageQuantizedMask);
        removeChannel(imageQuantized,4,size[0],size[1],2,imageQuantizedNoB);
        removeChannel(imageQuantizedMask,4,size[0],size[1],2,imageQuantizedMaskNoB);
        isMasked = viewer.isMasked();
        stateBKey = viewer.getStateBKey();
        if (!isMasked)
        {
            if (!stateBKey)        
                hpld.replaceLastTexture((unsigned char *)imageQuantized,size[0],size[1],4);
            else
                hpld.replaceLastTexture((unsigned char *)imageQuantizedNoB,size[0],size[1],4);
        }
        else
        {
            if (!stateBKey)
                hpld.replaceLastTexture((unsigned char *)imageQuantizedMask,size[0],size[1],4);
            else
                hpld.replaceLastTexture((unsigned char *)imageQuantizedMaskNoB,size[0],size[1],4);
        }
    }
    else
    {
        if (isMasked!=viewer.isMasked() || stateBKey!=viewer.getStateBKey())
        {            
            isMasked = viewer.isMasked();
            stateBKey = viewer.getStateBKey();
            if (!isMasked)
            {
                if (!stateBKey)        
                    hpld.replaceLastTexture((unsigned char *)imageQuantized,size[0],size[1],4);
                else
                    hpld.replaceLastTexture((unsigned char *)imageQuantizedNoB,size[0],size[1],4);
            }
            else
            {
                if (!stateBKey)
                    hpld.replaceLastTexture((unsigned char *)imageQuantizedMask,size[0],size[1],4);
                else
                    hpld.replaceLastTexture((unsigned char *)imageQuantizedMaskNoB,size[0],size[1],4);
            }
        }
    }
    render(&viewer);
  }

  /* clean exit; all okay */
  Hale::done();
  airMopOkay(mop);

  return 0;
}
