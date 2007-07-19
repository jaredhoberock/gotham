/*! \file Transform.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Transform class.
 */

#include "Transform.h"
#include "BoundingBox.h"
#include "Ray.h"

Transform
  ::Transform(const float m00, const float m01, const float m02, const float m03,
              const float m10, const float m11, const float m12, const float m13,
              const float m20, const float m21, const float m22, const float m23,
              const float m30, const float m31, const float m32, const float m33)
{
  gpcpu::float4x4 m;
  m(0,0) = m00; m(0,1) = m01; m(0,2) = m02; m(0,3) = m03;
  m(1,0) = m10; m(1,1) = m11; m(1,2) = m12; m(1,3) = m13;
  m(2,0) = m20; m(2,1) = m21; m(2,2) = m22; m(2,3) = m23;
  m(3,0) = m30; m(3,1) = m31; m(3,2) = m32; m(3,3) = m33;

  gpcpu::float4x4 invM = m.inverse();

  set(m, invM);
} // end Transform::Transform()

Point Transform
  ::transformPoint(const gpcpu::float4x4 &m,
                   const Point &p)
{
  float x = m(0,0)*p[0] + m(0,1)*p[1] + m(0,2)*p[2] + m(0,3);
  float y = m(1,0)*p[0] + m(1,1)*p[1] + m(1,2)*p[2] + m(1,3);
  float z = m(2,0)*p[0] + m(2,1)*p[1] + m(2,2)*p[2] + m(2,3);
  float w = m(3,0)*p[0] + m(3,1)*p[1] + m(3,2)*p[2] + m(3,3);

  if(w == 1.0f) return Point(x,y,z);

  return Point(x/w,y/w,z/w);
} // end Transform::transformPoint()

Point Transform
  ::operator()(const Point &p) const
{
  return transformPoint(Parent::first, p);
} // end Transform::operator()()

Transform Transform
  ::operator*(const Transform &rhs) const
{
  gpcpu::float4x4 product = first * rhs.first;
  return Transform(product);
} // end Transform::operator*()

Point Transform
  ::inverseTransform(const Point &p) const
{
  return transformPoint(Parent::second, p);
} // end Transform::inverseTransform()

Vector3 Transform
  ::inverseTransform(const Vector3 &v) const
{
  return transformVector(Parent::second, v);
} // end Transform::inverseTransform()

Normal Transform
  ::inverseTransform(const Normal &n) const
{
  // since transformNormal() wants the inverse of
  // the Transform to apply, send the first matrix, not the second
  return transformNormal(Parent::first, n);
} // end Transform::inverseTransform()

Ray Transform
  ::operator()(const Ray &r) const
{
  Ray result = r;

  // first transform the anchor
  Point newAnchor = (*this)(r.getAnchor());
  Vector newDirection = (*this)(r.getDirection());
  result.setAnchor(newAnchor);
  result.setDirection(newDirection);

  return result;
} // end Transform::operator()()

void Transform
  ::inverseTransform(const Ray &r, Ray &xfrmd) const
{
  xfrmd = r;

  // first transform the anchor
  Point newAnchor = inverseTransform(xfrmd.getAnchor());
  Vector newDirection = inverseTransform(xfrmd.getDirection());
  xfrmd.setAnchor(newAnchor);
  xfrmd.setDirection(newDirection);
} // end Transform::inverseTransform()

BoundingBox Transform
  ::operator()(const BoundingBox &b) const
{
  Point m = b.getMinCorner();
  Point M = b.getMaxCorner();

  Point p1(m[0], m[1], M[2]);
  Point p2(m[0], M[1], m[2]);
  Point p3(m[0], M[1], M[2]);
  Point p4(M[0], m[1], m[2]);
  Point p5(M[0], m[1], M[2]);
  Point p6(M[0], M[1], m[2]);

  BoundingBox result;

  // transform each corner: not just min & max

  result.addPoint((*this)(m));
  result.addPoint((*this)(M));
  result.addPoint((*this)(p1));
  result.addPoint((*this)(p2));
  result.addPoint((*this)(p3));
  result.addPoint((*this)(p4));
  result.addPoint((*this)(p5));
  result.addPoint((*this)(p6));

  return result;
} // end Transform::operator()()

Vector3 Transform
  ::operator()(const Vector3 &v) const
{
  return transformVector(Parent::first, v);
} // end Transform::operator()()

Normal Transform
  ::operator()(const Normal &n) const
{
  // pass the inverse to the function
  return transformNormal(Parent::second, n);
} // end Transform::operator()()

Vector3 Transform
  ::transformVector(const gpcpu::float4x4 &m,
                    const Vector3 &v)
{
  float x = m(0,0)*v[0] + m(0,1)*v[1] + m(0,2)*v[2];
  float y = m(1,0)*v[0] + m(1,1)*v[1] + m(1,2)*v[2];
  float z = m(2,0)*v[0] + m(2,1)*v[1] + m(2,2)*v[2];

  return Vector3(x,y,z);
} // end Transform::transformVector()

Normal Transform
  ::transformNormal(const gpcpu::float4x4 &inv,
                    const Normal &n)
{
  float x = inv(0,0)*n[0] + inv(1,0)*n[1] + inv(2,0)*n[2];
  float y = inv(0,1)*n[0] + inv(1,1)*n[1] + inv(2,1)*n[2];
  float z = inv(0,2)*n[0] + inv(1,2)*n[1] + inv(2,2)*n[2];

  return Normal(x,y,z);
} // end Transform::transformNormal()

Transform Transform
  ::translate(const float dx,
              const float dy,
              const float dz)
{
  gpcpu::float4x4 xfrm = gpcpu::float4x4::identity();
  xfrm(0,3) = dx;
  xfrm(1,3) = dy;
  xfrm(2,3) = dz;

  return Transform(xfrm);
} // end Transform::translate()

#define RADIANS(x) (x * (PI/180.0f))
#ifndef PI
#define PI 3.14159265f
#endif // PI

Transform Transform
  ::rotate(const float degrees,
           const float rx,
           const float ry,
           const float rz)
{
  gpcpu::float4x4 A;
  float length = sqrt((rx*rx) + (ry*ry) + (rz*rz));
  float a = rx / length;
  float b = ry / length;
  float c = rz / length;
  float aa = a * a;
  float bb = b * b;
  float cc = c * c;
  float sine = sin(RADIANS(-degrees));
  float cosine = cos(RADIANS(-degrees));
  float omcos = 1.0f - cosine;

  A(0,0) = aa + (1.0f - aa) * cosine;
  A(1,1) = bb + (1.0f - bb) * cosine;
  A(2,2) = cc + (1.0f - cc) * cosine;
  A(0,1) = a * b * omcos + c * sine;
  A(0,2) = a * c * omcos - b * sine;
  A(1,0) = a * b * omcos - c * sine;
  A(1,2) = b * c * omcos + a * sine;
  A(2,0) = a * c * omcos + b * sine;
  A(2,1) = b * c * omcos - a * sine;
  A(0,3) = A(1,3) = A(2,3) = A(3,0) = A(3,1) = A(3,2) = 0.0f;
  A(3,3) = 1.0f;
   
  return Transform(A);
} // end Transform::rotate()

Vector3 Transform
  ::rotateVector(const float degrees,
                 const float rx,
                 const float ry,
                 const float rz,
                 const Vector3 &v)
{
  gpcpu::float3x3 A;
  float length = sqrt((rx*rx) + (ry*ry) + (rz*rz));
  float a = rx / length;
  float b = ry / length;
  float c = rz / length;
  float aa = a * a;
  float bb = b * b;
  float cc = c * c;
  float sine = sin(RADIANS(-degrees));
  float cosine = cos(RADIANS(-degrees));
  float omcos = 1.0f - cosine;

  A(0,0) = aa + (1.0f - aa) * cosine;
  A(1,1) = bb + (1.0f - bb) * cosine;
  A(2,2) = cc + (1.0f - cc) * cosine;
  A(0,1) = a * b * omcos + c * sine;
  A(0,2) = a * c * omcos - b * sine;
  A(1,0) = a * b * omcos - c * sine;
  A(1,2) = b * c * omcos + a * sine;
  A(2,0) = a * c * omcos + b * sine;
  A(2,1) = b * c * omcos - a * sine;

  return A * v;
} // end Transform::rotateVector()

Transform Transform
  ::scale(const float sx,
          const float sy,
          const float sz)
{
  gpcpu::float4x4 xfrm = gpcpu::float4x4::identity();
  xfrm(0,0) = sx;
  xfrm(1,1) = sy;
  xfrm(2,2) = sz;

  return Transform(xfrm);
} // end Transform::scale()

