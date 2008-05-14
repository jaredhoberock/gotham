/*! \file UnitSquareToCone.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a uniform mapping of the unit square
 *         to vectors on a cone.
 */

#ifndef UNIT_SQUARE_TO_CONE_H
#define UNIT_SQUARE_TO_CONE_H

class UnitSquareToCone
{
  public:
    /*! This method maps points in the unit square
     *  to points uniformly distributed over a cone
     *  of directions (oriented toward +z)
     *  \param theta The "diameter" of the cone, in radians.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param p The mapping (u,v) -> to a point on the cone
     *           of directions is returned here.
     *  \param pdf The value of the distribution function over
     *             the cone of directions is returned here if
     *             this is not 0.
     */
    template<typename Real, typename Real3>
      static void evaluate(const Real &theta,
                           const Real &u,
                           const Real &v,
                           Real3 &p,
                           Real *pdf = 0);

    /*! This method evaluates the probability density
     *  function of this mapping at the given point on
     *  the cone of directions.
     *  \param p The point of interest on the cone of
     *           directions.
     *  \param theta The "diameter" of the cone, in radians.
     *  \return The value of the pdf at p.
     */
    template<typename Real3, typename Real>
      static Real evaluatePdf(const Real3 &p,
                              const Real &theta);
}; // end UnitSquareToCone

#include "UnitSquareToCone.inl"

#endif // UNIT_SQUARE_TO_CONE_H

