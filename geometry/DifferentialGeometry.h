/*! \file DifferentialGeometry.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class describing the differential geometric properties of a point on a surface.
 *         XXX clean up this interface
 */

#ifndef DIFFERENTIAL_GEOMETRY_H
#define DIFFERENTIAL_GEOMETRY_H

#include <gpcpu/Vector.h>
#include "Point.h"
#include "Vector.h"
#include "Normal.h"
#include "ParametricCoordinates.h"
class Surface;

/*! \class DifferentialGeometry
 *  \brief DifferentialGeometry encapsulates the description
 *         of the differential geometric properties of a surface
 *         at a particular Point.
 */
class DifferentialGeometry
{
  public:
    /*! Null constructor does nothing.
     */
    inline DifferentialGeometry(void);

    /*! Constructor accepts a Point, Point and Normal partials, ParametricCoordinates, and a Surface.
     *  \param p Sets mPoint.
     *  \param dpdu Sets mPointPartials[0].
     *  \param dpdv Sets mPointPartials[1].
     *  \param dndu Sets mNormalVectorPartials[0].
     *  \param dndv Sets mNormalVectorPartials[1].
     *  \param uv Sets mParametricCoordinates.
     *  \param s Sets mSurface.
     *  \note mNormalVector is computed and set as a side effect.
     */
    inline DifferentialGeometry(const Point &p,
                                const Vector3 &dpdu, const Vector3 &dpdv,
                                const Vector3 &dndu, const Vector3 &dndv,
                                const ParametricCoordinates &uv, const Surface *s);

    /*! Constructor accepts a Point, Normal, Point and Normal partials, ParametricCoordinates, and a Surface.
     *  \param p Sets mPoint.
     *  \param n Sets mNormalVector.
     *  \param dpdu Sets mPointPartials[0].
     *  \param dpdv Sets mPointPartials[1].
     *  \param dndu Sets mNormalVectorPartials[0].
     *  \param dndv Sets mNormalVectorPartials[1].
     *  \param uv Sets mParametricCoordinates.
     *  \param s Sets mSurface.
     *  \note Use this constructor when dpdu x dpdv is suspect, or you have an analytic Normal vector.
     */
    inline DifferentialGeometry(const Point &p, const Normal &n,
                                const Vector3 &dpdu, const Vector3 &dpdv,
                                const Vector3 &dndu, const Vector3 &dndv,
                                const ParametricCoordinates &uv, const Surface *s);

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

    /*! This method sets mSurface.
     *  \param s Sets mSurface.
     */
    inline void setSurface(const Surface *s);

    /*! This method returns mSurface.
     *  \return mSurface
     */
    inline const Surface *getSurface(void) const;

    /*! This method sets mPointPartials.
     *  \param dpdu Sets mPointPartials[0].
     *  \param dpdv Sets mPointPartials[1].
     */
    inline void setPointPartials(const Vector3 &dpdu, const Vector3 &dpdv);

    /*! This method returns a const pointer to mPointPartials.
     *  \return mPointPartials.
     */
    inline const Vector3 *getPointPartials(void) const;

    /*! This method returns a pointer to mPointPartials.
     *  \return mPointPartials.
     */
    inline Vector3 *getPointPartials(void);

    /*! This method sets mNormalVectorPartials.
     *  \param dndu Sets mNormalVectorPartials[0].
     *  \param dndv Sets mNormalVectorPartials[1].
     */
    inline void setNormalVectorPartials(const Vector3 &dndu, const Vector3 &dndv);

    /*! This method returns a const pointer to mNormalVectorPartials.
     *  \return mNormalVectorPartials.
     */
    inline const Vector3 *getNormalVectorPartials(void) const;

    /*! This method returns a pointer to mNormalVectorPartials.
     *  \return mNormalVectorPartials.
     */
    inline Vector3 *getNormalVectorPartials(void);

  protected:
    /*! The Point on mSurface this DifferentialGeometry object describes.
     *  This Point is in world coordinates.
     */
    Point mPoint;

    /*! The Normal vector on Surface mSurface at Point mPoint.
     *  This Normal is assumed to be normalized.
     */
    Normal mNormal;

    /*! The ParametricCoordinates of Surface mSurface at Point mPoint.
     */
    ParametricCoordinates mParametricCoordinates;

    /*! The surface whose geometry we are describing.
     */
    const Surface *mSurface;

    /*! The partial derivatives of mPoint with respect to
     *  mParametricCoordinates[0] and mParametricCoordinates[1],
     *  respectively.
     */
    Vector3 mPointPartials[2];

    /*! The partial derivatives of mNormalVector with respect
     *  to mParametricCoordinates[0] and mParametricCoordinates[1],
     *  respectively.
     */
    Vector3 mNormalVectorPartials[2];
}; // end class DifferentialGeometry

#include "DifferentialGeometry.inl"

#endif // DIFFERENTIAL_GEOMETRY_H

