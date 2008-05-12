/*! \file DifferentialGeometryArrayBase.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an array
 *         view of differential geometry vectors
 *         for flexibility.
 */

#pragma once

template<typename P3,
         typename V3,
         typename P2,
         typename N3>
  struct DifferentialGeometryArrayBase
{
  public:
    typedef P3 Point;
    typedef V3 Vector;
    typedef P2 ParametricCoordinates;
    typedef N3 Normal;

    /*! operator+=() increments each pointer.
     *  \param diff The increment size.
     *  \return *this
     */
    inline DifferentialGeometryArrayBase &operator+=(const ptrdiff_t diff);

    /*! operator+ adds to each pointer.
     *  \param diff The increment size.
     *  \return (*this) + diff
     */
    inline DifferentialGeometryArrayBase operator+(const ptrdiff_t diff) const;

    /*! The points array
     */
    Point *mPoints;

    /*! The normals array
     */
    Normal *mNormals;

    /*! The tangents array
     */
    Vector *mTangents;

    /*! The binormals array
     */
    Vector *mBinormals;

    /*! The parametric coordinates array
     */
    ParametricCoordinates *mParametricCoordinates;

    /*! The surface area array.
     */
    float *mSurfaceAreas;

    /*! The inverse surface area array.
     */
    float *mInverseSurfaceAreas;

    /*! The array of partial derivatives of mPoint with respect
     *  to mParametricCoordinates.x
     */
    Vector *mDPDUs;

    /*! The array of partial derivaties of mPoint with respect
     *  to mParametricCoordinates.y
     */
    Vector *mDPDVs;

    /*! The array of partial derivatives of mNormal with respect
     *  to mParametricCoordinates.x
     */
    Vector *mDNDUs;

    /*! The array of partial derivatives of mNormal with respect
     *  to mParametricCoordinates.y
     */
    Vector *mDNDVs;
}; // end DifferentialGeometryArrayBase

#include "DifferentialGeometryArrayBase.inl"

