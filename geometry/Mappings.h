/*! \file Mappings.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class
 *         which provides functions for mapping
 *         the unit square to various familiar
 *         manifolds.
 */

#ifndef MAPPINGS_H
#define MAPPINGS_H

#include "Vector.h"

class Mappings
{
  public:
    /*! This method maps a point in the unit square
     *  to a point on the unit hemisphere situated around
     *  the z-axis.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param xAxis A unit vector parallel to the x-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param yAxis A unit vector parallel to the y-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param zAxis A unit vector parallel to the z-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param w The point on the unit hemisphere is returned here.
     *  \param pdf The value of the pdf at p is returned here.
     *  \note xAxis, yAxis, and zAxis are assumed to define an
     *        orthonormal basis.
     */
    static void unitSquareToHemisphere(const float u0,
                                       const float u1,
                                       const Vector3 &xAxis,
                                       const Vector3 &yAxis,
                                       const Vector3 &zAxis,
                                       Vector3 &w,
                                       float &pdf);

    /*! This method maps a point in the unit square to a point
     *  on the cosine-weighted unit hemisphere situated around
     *  a vector identified with the z-axis.
     *  \param u0 A real number in [0,1).
     *  \param u1 A second real number in [0,1).
     *  \param xAxis A unit vector parallel to the x-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param yAxis A unit vector parallel to the y-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param zAxis A unit vector parallel to the z-axis of
     *               defining the frame of the hemisphere of interest.
     *  \param w The point on the unit hemisphere is returned here.
     *  \param pdf The value of the pdf at p is returned here.
     *  \note xAxis, yAxis, and zAxis are assumed to define an
     *        orthonormal basis.
     */
    static void unitSquareToCosineHemisphere(const float u0,
                                             const float u1,
                                             const Vector3 &xAxis,
                                             const Vector3 &yAxis,
                                             const Vector3 &zAxis,
                                             Vector3 &w,
                                             float &pdf);

}; // end Mappings

#endif // MAPPINGS_H

