/*! \file UnitSquareToDisk.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class for mapping
 *         from the unit square to the unit disk.
 */

#ifndef UNIT_SQUARE_TO_DISK_H
#define UNIT_SQUARE_TO_DISK_H

class UnitSquareToDisk
{
  public:
    /*! This method maps points in the unit square to
     *  points uniformly distributed over the unit disk.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param p The mapping (u,v) -> to a point on the unit disk
     *           is returned here.
     *  \param pdf The value of the distribution function
     *             over the hemisphere is returned here if
     *             this is not 0.
     */
    template<typename Real, typename Real2>
      static void evaluate(const Real &u,
                           const Real &v,
                           Real2 &p,
                           Real *pdf = 0);

    /*! This method maps points in the unit square to
     *  points uniformly distributed over the unit disk.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param x The x-coordinate of the point on the unit disk
     *           is returned here.
     *  \param y The y-coordinate of the point on the unit disk
     *           is returned here.
     *  \param pdf The value of the distribution function
     *             over the hemisphere is returned here.
     */
    template<typename Real>
      static void evaluate(const Real &u, const Real &v,
                           Real &x, Real &y,
                           Real &pdf);

    /*! This method maps points in the unit square to
     *  points uniformly distributed over the unit disk.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param x The x-coordinate of the point on the unit disk
     *           is returned here.
     *  \param y The y-coordinate of the point on the unit disk
     *           is returned here.
     */
    template<typename Real>
      static void evaluate(const Real &u, const Real &v,
                           Real &x, Real &y);

    /*! This method provides the inverse mapping from
     *  a point on the unit disk to a point on the unit
     *  square.
     *  \param p The point on the unit disk.
     *  \param u The first coordinate of the corresponding
     *           point on the unit square is returned here.
     *  \param v The second coordinate of the corresponding
     *           point on the unit square is returned here.
     */
    template<typename Real2, typename Real>
      static void inverse(const Real2 &p,
                          Real &u,
                          Real &v);
}; // end class UnitSquareToDisk

#include "UnitSquareToDisk.inl"

#endif // UNIT_SQUARE_TO_DISK_H

