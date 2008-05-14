/*! \file UnitSquareToHemisphere.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class for mapping
 *         from the unit square to the unit hemisphere.
 */

#ifndef UNIT_SQUARE_TO_HEMISPHERE_H
#define UNIT_SQUARE_TO_HEMISPHERE_H

class UnitSquareToHemisphere
{
  public:
    /*! This method maps points in the unit square
     *  to points uniformly distributed over the
     *  unit hemisphere (oriented toward +z)
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param p The mapping (u,v) -> to a point on the unit hemisphere
     *           is returned here.
     *  \param pdf The value of the distribution function
     *             over the hemisphere is returned here if
     *             this is not 0.
     */
    template<typename Real, typename Real3>
      static void evaluate(const Real &u,
                           const Real &v,
                           Real3 &p,
                           Real *pdf = 0);

    /*! This method provides the inverse mapping from
     *  a point on the hemisphere to a point on the
     *  unit square.
     *  \param p The point on the unit hemisphere.
     *  \param u The first coordinate of the corresponding
     *           point on the unit square is returned here.
     *  \param v The second coordinate of the corresponding
     *           point on the unit square is returned here.
     */
    template<typename Real3, typename Real>
      static void inverse(const Real3 &p,
                          Real &u,
                          Real &v);

    /*! This method evaluates the probability density
     *  function of this mapping at the given point
     *  on the unit hemisphere.
     *  \param p The point of interest on the
     *           hemisphere.
     *  \return The value of the pdf at p.
     */
    template<typename Real, typename Real3>
      static Real evaluatePdf(const Real3 &p);
}; // end UnitSquareToHemisphere

#include "UnitSquareToHemisphere.inl"

#endif // UNIT_SQUARE_TO_HEMISPHERE_H

