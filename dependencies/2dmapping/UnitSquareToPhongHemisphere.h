/*! \file UnitSquareToPhongHemisphere.h
 *  \author Jared Hoberock
 *  \brief Defines the interface a mapping from the unit square to the
 *         Phong distribution over the hemisphere.
 */

#ifndef UNIT_SQUARE_TO_PHONG_HEMISPHERE_H
#define UNIT_SQUARE_TO_PHONG_HEMISPHERE_H

template<typename Real, typename Real3>
  class UnitSquareToPhongHemisphere
{
  public:
    /*! This method maps points in the unit square
     *  to points with a Phong distribution centered
     *  about the unit hemisphere with normal vector
     *  pointing towards +z.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param k The exponent of the distribution.
     *  \param p The mapping (u,v) -> to a point on the phong hemisphere
     *           is returned here.
     *  \param pdf The value of the distribution function over the hemisphere
     *             is returned here if this is not 0.
     */
    inline static void evaluate(const Real &u,
                                const Real &v,
                                const Real &k,
                                Real3 &p,
                                Real *pdf = 0);

    /*! This method evaluates the probability density
     *  function of this mapping at the given point
     *  on the Phong-weighted unit hemisphere.
     *  \param p The point of interest on the Phong-weighted
     *           hemisphere.
     *  \param k The exponent of the distribution.
     *  \return The value of the pdf at p.
     */
    inline static Real evaluatePdf(const Real3 &p,
                                   const Real &k);

    /*! This method provides the inverse mapping from
     *  a point on the Phong-weighted hemisphere to a
     *  point on the unit square.
     *  \param p The point on the unit hemisphere.
     *  \param k The exponent of the distribution.
     *  \param u The first coordinate of the corresponding
     *           point on the unit square is returned here.
     *  \param v The second coordinate of the corresponding
     *           point on the unit square is returned here.
     */
    inline static void inverse(const Real3 &p,
                               const Real &k,
                               Real &u,
                               Real &v);
}; // end UnitSquareToPhongHemisphere

#include "UnitSquareToPhongHemisphere.inl"

#endif // UNIT_SQUARE_TO_PHONG_HEMISPHERE_H

