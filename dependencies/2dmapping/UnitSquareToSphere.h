/*! \file UnitSquareToSphere.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class for mapping
 *         from the unit square to the unit sphere.
 */

#ifndef UNIT_SQUARE_TO_SPHERE_H
#define UNIT_SQUARE_TO_SPHERE_H

class UnitSquareToSphere
{
  public:
    /*! This method maps points in the unit square to
     *  points uniformly distributed over the unit sphere.
     *  \param u The first coordinate of the point in the
     *           unit square.
     *  \param v The second coordinate of the point in the
     *           unit square.
     *  \param p The mapping (u,v) -> to a point on the
     *           unit sphere is returned here.
     *  \param pdf The value of the distribution function
     *             over the sphere is returned here if this
     *             is not 0.
     */
    template<typename Real, typename Real3>
      static void evaluate(const Real &u,
                           const Real &v,
                           Real3 &p,
                           Real *pdf = 0);

    /*! This method evaluates the probability density
     *  function of this mapping at the given point
     *  on the unit sphere.
     *  \param p The point of interest on the sphere.
     *  \return The value of the pdf at p.
     *  \note That the pdf is constant over this mapping,
     *        so the value of p is ignored (but still
     *        assumed to actually lie on the sphere).
     */
    template<typename Real, typename Real3>
      static Real evaluatePdf(const Real3 &p);
}; // end class UnitSquareToSphere

#include "UnitSquareToSphere.inl"

#endif // UNIT_SQUARE_TO_SPHERE_H

