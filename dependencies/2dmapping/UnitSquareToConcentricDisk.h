/*! \file UnitSquareToConcentricDisk.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class for mapping
 *         from the unit square to the concentric unit disk.
 */

#ifndef UNIT_SQUARE_TO_CONCENTRIC_DISK_H
#define UNIT_SQUARE_TO_CONCENTRIC_DISK_H

template<typename Real, typename Real2>
  class UnitSquareToConcentricDisk
{
  public:
    /*! This method maps points in the unit square
     *  to points uniformly concentricly distributed
     *  over the unit disk.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param p The mapping (u,v) -> to a point on the unit disk
     *           is returned here.
     *  \param pdf The value of the distribution function
     *             over the disk is returned here if
     *             this is not 0.
     */
    inline static void evaluate(const Real &u,
                                const Real &v,
                                Real2 &p,
                                Real *pdf = 0);

    /*! This method maps points in the unit square
     *  to points uniformly concentricly distributed
     *  over the unit disk.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param x The x-coordinate of the point in the unit concentric disk 
     *           is returned here.
     *  \param y The y-coordinate of the point in the unit concentric disk 
     *           is returned here.
     *  \param z The z-coordinate of the point in the unit concentric disk 
     *           is returned here.
     *  \param pdf The value of the distribution function
     *             over the disk is returned here.
     */
    inline static void evaluate(const Real &u, const Real &v,
                                Real &x, Real &y,
                                Real &pdf);

    /*! This method maps points in the unit square
     *  to points uniformly concentricly distributed
     *  over the unit disk.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param x The x-coordinate of the point in the unit concentric disk 
     *           is returned here.
     *  \param y The y-coordinate of the point in the unit concentric disk 
     *           is returned here.
     *  \param z The z-coordinate of the point in the unit concentric disk 
     *           is returned here.
     */
    inline static void evaluate(const Real &u, const Real &v,
                                Real &x, Real &y);

    /*! This method provides the inverse mapping from
     *  a point on the concentric disk to a
     *  point on the unit square.
     *  \param p The point on the unit disk.
     *  \param u The first coordinate of the corresponding
     *           point on the unit square is returned here.
     *  \param v The second coordinate of the corresponding
     *           point on the unit square is returned here.
     */
    inline static void inverse(const Real2 &p,
                               Real &u,
                               Real &v);
}; // end UnitSquareToConcentricDisk

#include "UnitSquareToConcentricDisk.inl"

#endif // UNIT_SQUARE_TO_CONCENTRIC_DISK_H

