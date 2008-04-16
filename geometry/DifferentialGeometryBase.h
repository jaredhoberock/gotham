/*! \file DifferentialGeometryBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class describing the differential geometric properties of a point on a surface.
 *         XXX clean up this interface
 */

#ifndef DIFFERENTIAL_GEOMETRY_BASE_H
#define DIFFERENTIAL_GEOMETRY_BASE_H

/*! \class DifferentialGeometryBase
 *  \brief DifferentialGeometryBase encapsulates the description
 *         of the differential geometric properties of a surface
 *         at a particular Point.
 *  \note This is made a template so we can substitute CUDA vector
 *        types.  Thanks, CUDA!
 */
template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  class DifferentialGeometryBase
{
  public:
    typedef P3 Point;
    typedef V3 Vector;
    typedef P2 ParametricCoordinates;
    typedef N3 Normal;

    /*! Null constructor does nothing.
     */
    inline DifferentialGeometryBase(void);

    /*! Constructor accepts a Point, Point and Normal partials, ParametricCoordinates, and a Surface.
     *  \param p Sets mPoint.
     *  \param dpdu Sets mPointPartials[0].
     *  \param dpdv Sets mPointPartials[1].
     *  \param dndu Sets mNormalVectorPartials[0].
     *  \param dndv Sets mNormalVectorPartials[1].
     *  \param uv Sets mParametricCoordinates.
     *  \param a Sets mSurfaceArea.
     *  \param invA Sets mInverseSurfaceArea.
     *  \note mNormalVector is computed and set as a side effect.
     */
    inline DifferentialGeometryBase(const Point &p,
                                    const Vector &dpdu, const Vector &dpdv,
                                    const Vector &dndu, const Vector &dndv,
                                    const ParametricCoordinates &uv,
                                    const float a,
                                    const float invA);

    /*! Constructor accepts a Point, Normal, Point and Normal partials, ParametricCoordinates, and a Surface.
     *  \param p Sets mPoint.
     *  \param n Sets mNormalVector.
     *  \param dpdu Sets mPointPartials[0].
     *  \param dpdv Sets mPointPartials[1].
     *  \param dndu Sets mNormalVectorPartials[0].
     *  \param dndv Sets mNormalVectorPartials[1].
     *  \param uv Sets mParametricCoordinates.
     *  \param a Sets mSurfaceArea.
     *  \param invA Sets mInverseSurfaceArea.
     *  \note Use this constructor when dpdu x dpdv is suspect, or you have an analytic Normal vector.
     */
    inline DifferentialGeometryBase(const Point &p, const Normal &n,
                                    const Vector &dpdu, const Vector &dpdv,
                                    const Vector &dndu, const Vector &dndv,
                                    const ParametricCoordinates &uv,
                                    const float a,
                                    const float invA);

    /*! This method sets mPoint.
     *  \param p Sets mPoint.
     */
    inline void setPoint(const Point &p);

    /*! This method returns a reference to mPoint.
     *  \return mPoint.
     */
    inline Point &getPoint(void);

    /*! This method returns a const reference to mPoint.
     *  \return mPoint.
     */
    inline const Point &getPoint(void) const;

    /*! This method sets mNormal.
     *  \param n Sets mNormal.
     */
    inline void setNormal(const Normal &n);

    /*! This method returns a reference to mNormalVector.
     *  \return mNormalVector.
     */
    inline Normal &getNormal(void);

    /*! This method returns a const reference to mNormalVector.
     *  \return mNormalVector.
     */
    inline const Normal &getNormal(void) const;

    /*! This method sets mParametericCoordinates.
     *  \param uv Sets mParametricCoordinates.
     */
    inline void setParametricCoordinates(const ParametricCoordinates &uv);

    /*! This method returns a const reference to mParametricCoordinates.
     *  \return mParametricCoordinates.
     */
    inline const ParametricCoordinates &getParametricCoordinates(void) const;

    /*! This method returns a reference to mParametricCoordinates.
     *  \return mParametricCoordinates.
     */
    inline ParametricCoordinates &getParametricCoordinates(void);

    /*! This method sets mSurfaceArea.
     *  \param a Sets mSurfaceArea.
     */
    inline void setSurfaceArea(const float a);

    /*! This method returns mSurfaceArea.
     *  \return mSurfaceArea.
     */
    inline float getSurfaceArea(void) const;

    /*! This method sets mInverseSurfaceArea.
     *  \param invA Sets mInverseSurfaceArea.
     */
    inline void setInverseSurfaceArea(const float invA);

    /*! This method returns mInverseSurfaceArea.
     *  \return mInverseSurfaceArea.
     */
    inline float getInverseSurfaceArea(void) const;

    /*! This method sets mPointPartials.
     *  \param dpdu Sets mPointPartials[0].
     *  \param dpdv Sets mPointPartials[1].
     */
    inline void setPointPartials(const Vector &dpdu, const Vector &dpdv);

    /*! This method returns a const pointer to mPointPartials.
     *  \return mPointPartials.
     */
    inline const Vector *getPointPartials(void) const;

    /*! This method returns a pointer to mPointPartials.
     *  \return mPointPartials.
     */
    inline Vector *getPointPartials(void);

    /*! This method sets mNormalVectorPartials.
     *  \param dndu Sets mNormalVectorPartials[0].
     *  \param dndv Sets mNormalVectorPartials[1].
     */
    inline void setNormalVectorPartials(const Vector &dndu, const Vector &dndv);

    /*! This method returns a const pointer to mNormalVectorPartials.
     *  \return mNormalVectorPartials.
     */
    inline const Vector *getNormalVectorPartials(void) const;

    /*! This method returns a pointer to mNormalVectorPartials.
     *  \return mNormalVectorPartials.
     */
    inline Vector *getNormalVectorPartials(void);

    /*! This method sets mTangent.
     *  \param t Sets mTangent.
     */
    inline void setTangent(const Vector &t);

    /*! This method returns mTangent.
     *  \return mTangent.
     */
    inline const Vector &getTangent(void) const;

    /*! This method sets mBinormal.
     *  \param b Sets mBinormal.
     */
    inline void setBinormal(const Vector &b);

    /*! This method returns mBinormal.
     *  \return mBinormal.
     */
    inline const Vector &getBinormal(void) const;

  protected:
    /*! The Point on mSurface this DifferentialGeometry object describes.
     *  This Point is in world coordinates.
     */
    Point mPoint;

    /*! The Normal vector on Surface mSurface at Point mPoint.
     *  This Normal is assumed to be normalized.
     */
    Normal mNormal;

    /*! The tangent vector is a unit vector which points along mPointPartials[0].
     */
    Vector mTangent;

    /*! The binormal vector is a unit vector which is orthogonal to
     *  to mNormal and mTangent.
     */
    Vector mBinormal;

    /*! The ParametricCoordinates of Surface mSurface at Point mPoint.
     */
    ParametricCoordinates mParametricCoordinates;

    /*! The area of the surface whose geometry we are describing.
     */
    float mSurfaceArea;

    /*! The inverse of the surface area of the surface whose geometry
     *  we are describing.
     */
    float mInverseSurfaceArea;

    /*! The partial derivatives of mPoint with respect to
     *  mParametricCoordinates[0] and mParametricCoordinates[1],
     *  respectively.
     */
    Vector mPointPartials[2];

    /*! The partial derivatives of mNormalVector with respect
     *  to mParametricCoordinates[0] and mParametricCoordinates[1],
     *  respectively.
     */
    Vector mNormalVectorPartials[2];
}; // end class DifferentialGeometryBase

#include "DifferentialGeometryBase.inl"

#endif // DIFFERENTIAL_GEOMETRY_BASE_H

