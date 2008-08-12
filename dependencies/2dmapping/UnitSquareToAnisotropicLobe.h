/*! \file UnitSquareToAnisotropicLobe.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a mapping from the unit square to the
 *         Ashihkmin-Shirley anisotropic distribution over the hemisphere.
 */

#ifndef UNIT_SQUARE_TO_ANISOTROPIC_LOBE_H
#define UNIT_SQUARE_TO_ANISOTROPIC_LOBE_H

template<typename Real, typename Real3>
  class UnitSquareToAnisotropicLobe
{
  public:
    /*! This method maps points in the unit square
     *  to points with a anisotropic distribution centered
     *  about the unit hemisphere with normal vector
     *  pointing towards +z.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param nu The first exponent of the distribution.
     *  \param nv The second exponent of the distribution.
     *  \param p The mapping (u,v) -> to a point on the phong hemisphere
     *           is returned here.
     *  \param pdf The value of the distribution function over the hemisphere
     *             is returned here if this is not 0.
     */
    inline static void evaluate(Real u,
                                Real v,
                                Real nu,
                                Real nv,
                                Real3 &p,
                                Real *pdf = 0);

    /*! This method evaluates the probability density
     *  function of this mapping at the given point
     *  on the anisotropic-weighted unit hemisphere.
     *  \param p The point of interest on the anisotropic-weighted
     *           hemisphere.
     *  \param nu The first exponent of the distribution.
     *  \param nv The second exponent of the distribution.
     *  \return The value of the pdf at p.
     */
    inline static Real evaluatePdf(const Real3 &p,
                                   const Real &nu,
                                   const Real &nv);

    inline static void sampleFirstQuadrant(const Real &u,
                                           const Real &v,
                                           const Real &nu,
                                           const Real &nv,
                                           Real &phi,
                                           Real &costheta);
}; // end UnitSquareToAnisotropicLobe

#include "UnitSquareToAnisotropicLobe.inl"

#endif // UNIT_SQUARE_TO_ANISOTROPIC_LOBE_H

