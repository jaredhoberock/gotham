/*! \file Transform.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class abstracting an affine transformation.
 */

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "../include/detail/gpcpu/floatmxn.h"
#include "../include/detail/Vector.h"
#include "../include/detail/Normal.h"
#include "../include/detail/Point.h"

class Ray;
class BoundingBox;

/*! \class Transform
 *  \brief A Transform is a pair of float4x4s with extra methods for transforming points, normals, Rays, etc.
 *         The first element of the pair is the forward transformation matrix, the second element of the pair is the
 *         reverse transformation matrix.
 */
class Transform
  : public std::pair<gpcpu::float4x4,gpcpu::float4x4>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef std::pair<gpcpu::float4x4,gpcpu::float4x4> Parent;

    /*! Null constructor calls the Parent.
     */
    inline Transform(void);

    /*! Constructor accepts a 4x4 matrix.
     */
    Transform(const float m00, const float m01, const float m02, const float m03,
              const float m10, const float m11, const float m12, const float m13,
              const float m20, const float m21, const float m22, const float m23,
              const float m30, const float m31, const float m32, const float m33);

    /*! Constructor accepts a float4x4.
     *  \param xfrm Sets this Transform.
     */
    inline Transform(const gpcpu::float4x4 &xfrm);

    /*! This method returns the inverse of this Transform.
     *  \param inv The inverse of this Transform is returned here.
     */
    inline void getInverse(Transform &inv) const;

    /*! This method sets this Transform given a matrix to use as the forward transformation and a matrix to use as the inverse
     *  transformation.
     *  \param xfrm The matrix representing the forward transformation.
     *  \param inv The matrix which is the inverse of xfrm.
     *  \note If inv is not actually the inverse of xfrm, it's your fault when it stops working.
     */
    inline void set(const gpcpu::float4x4 &xfrm, const gpcpu::float4x4 &inv);

    /*! This method transforms a Point by applying this Transform.
     *  \param p The Point to transform.
     *  \return p transformed.
     */
    Point operator()(const Point &p) const;

    /*! This method transforms a Point by applying this Transform's inverse.
     *  \param p The Point to transform.
     *  \return p transformed by this Transform's inverse.
     */
    Point inverseTransform(const Point &p) const;

    /*! This method transforms a vector.
     *  \param v The vector to transform.
     *  \return v transformed.
     */
    Vector operator()(const Vector &v) const;

    /*! This method transforms a Normal.
     *  \param n The Normal to transform.
     *  \return n transformed.
     */
    Normal operator()(const Normal &n) const;

    /*! This method transforms a Normal by applying this Transform's inverse.
     *  \param n The Normal to transform.
     *  \return n transformed by this Transform's inverse.
     */
    Normal inverseTransform(const Normal &n) const;

    /*! This method transforms a vector by applying this Transform's inverse.
     *  \param v The vector to transform.
     *  \return v transformed by this Transform's inverse.
     */
    Vector inverseTransform(const Vector &v) const;

    /*! This method transforms a BoundingBox by applying this Transform to its min/max corners.
     *  \param b The BoundingBox to transform.
     *  \return b transformed.
     */
    BoundingBox operator()(const BoundingBox &b) const;

    /*! This method transforms a Ray by applying this Transform to its anchor and direction.
     *  \param r The Ray to transform.
     *  \return r transformed.
     */
    Ray operator()(const Ray &r) const;

    /*! This method transforms a Ray by applying this Transform's inverse to its anchor and direction.
     *  \param r The Ray to transform.
     *  \param xfrm r transformed by this Transform's inverse is returned here.
     */
    void inverseTransform(const Ray &r, Ray &xfrmd) const;

    /*! This static method returns a Transform which represents the identity transformation.
     *  \return The identity matrix Transform.
     */
    inline static Transform identity(void);

    /*! This static method returns a Transform which represents a translation by the given vector.
     *  \param dx The x coordinate of the translation vector.
     *  \param dy The y coordinate of the translation vector.
     *  \param dz The z coordinate of the translation vector.
     *  \return The Transform representing the translation (dx,dy,dz).
     */
    static Transform translate(const float dx, const float dy, const float dz);

    /*! This static method returns a Transform which represents a counterclockwise rotation by the given degrees
     *  around the given vector.
     *  \param degrees How far to rotate, in degrees.
     *  \param rx The x coordinate of the rotation axis.
     *  \param ry The y coordinate of the rotation axis.
     *  \param rz The z coordinate of the rotation axis.
     *  \return The Transform representing the rotation of degrees around (rx,ry,rz).
     */
    static Transform rotate(const float degrees, const float rx, const float ry, const float rz);

    /*! This static method rotates a vector around a given axis.
     *  \param degrees How far to rotate, in degrees.
     *  \param rx The x coordinate of the rotation axis.
     *  \param ry The y coordinate of the rotation axis.
     *  \param rz The z coordinate of the rotation axis.
     *  \param v The vector to rotate.
     *  \return v rotated around (rx,ry,rz).
     */
    static Vector rotateVector(const float degrees, const float rx, const float ry, const float rz,
                               const Vector &v);

    /*! This static method returns a Transform which represents a scaling by the given scale factors.
     *  \param sx Scaling in x.
     *  \param sy Scaling in y.
     *  \param sz Scaling in z.
     *  \return The Transform representing a scaling of (sx, sy, sz).
     */
    static Transform scale(const float sx, const float sy, const float sz);

    /*! Multiplication operator returns the product of this Transform with another.
     *  \param rhs The right hand side of the multiplication.
     *  \return (*this)*rhs
     */
    Transform operator*(const Transform &rhs) const;

  protected:
    /*! This constructor creates a Transform given a matrix and its inverse.
     *  \param xfrm The matrix representing the forward transformation.
     *  \param inv The matrix which is the inverse of xfrm.
     *  \note If inv is not actually the inverse of xfrm, it's your fault when it stops working.
     */
    inline Transform(const gpcpu::float4x4 &xfrm, const gpcpu::float4x4 &inv);

    /*! This static method transforms a Point by a float4x4.
     *  \param m The float4x4 describing the transformation.
     *  \param p The Point to transform.
     *  \return p transformed by xfrm.
     */
    static Point transformPoint(const gpcpu::float4x4 &m, const Point &p);

    /*! This static method transform a vector by a float4x4.
     *  \param m The float4x4 describing the transformation.
     *  \param v The vector to transform.
     *  \return v transformed by xfrm.
     */
    static Vector transformVector(const gpcpu::float4x4 &m, const Vector &v);

    /*! This static method transforms a normal by a float4x4.
     *  \param m The float4x4 describing the inverse of the transformation.
     *  \param n The normal to transform.
     *  \return n transformed by xfrm.
     */
    static Normal transformNormal(const gpcpu::float4x4 &inv, const Normal &n);
}; // end class Transform

#include "Transform.inl"

#endif // TRANSFORM_H

