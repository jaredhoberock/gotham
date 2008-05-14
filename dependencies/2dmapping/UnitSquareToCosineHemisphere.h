/*! \file UnitSquareToCosineHemisphere.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class for mapping
 *         from the unit square to the cosine-weighted
 *         unit hemisphere.
 */

#ifndef UNIT_SQUARE_TO_COSINE_HEMISPHERE_H
#define UNIT_SQUARE_TO_COSINE_HEMISPHERE_H

template<typename Real, typename Real3>
  class UnitSquareToCosineHemisphere
{
  public:
    /*! This method maps points in the unit square
     *  to points uniformly distributed over the
     *  unit hemisphere according to a cosine distribution.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param p The mapping (u,v) -> to a point on the unit hemisphere
     *           is returned here.
     *  \param pdf The value of the distribution function
     *             over the hemisphere is returned here if
     *             this is not 0.
     */
    inline static void evaluate(const Real &u,
                                const Real &v,
                                Real3 &p,
                                Real *pdf = 0);

    /*! This method maps points in the unit square
     *  to points distributed over the unit hemisphere
     *  according to a cosine distribution.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param x The x-coordinate of the point on the hemisphere is
     *           returned here.
     *  \param y The y-coordinate of the point on the hemisphere is
     *           returned here.
     *  \param z The z-coordinate of the point on the hemisphere is *           returned here.
     *  \param pdf The value of the distribution function over the
     *             hemisphere is returned here.
     */
    static inline void evaluate(const Real &u, const Real &v,
                                Real &x, Real &y, Real &z,
                                Real &pdf);

    /*! This method evaluates the probability density
     *  function of this mapping at the given point
     *  on the cosine-weighted unit hemisphere.
     *  \param p The point of interest on the
     *           cosine-weighted hemisphere.
     *  \return The value of the pdf at p.
     */
    inline static Real evaluatePdf(const Real3 &p);

    /*! This method evaluates the probability density
     *  function of this mapping at the given point
     *  on the cosine-weighted unit hemisphere.
     *  \param x The x-coordinate of the point on the
     *           hemisphere.
     *  \param y The y-coordinate of the point on the
     *           hemisphere.
     *  \param z The z-coordinate of the point on the
     *           hemisphere.
     *  \return The value of the pdf at (x,y,z).
     */
    inline static Real evaluatePdf(const Real &x,
                                   const Real &y,
                                   const Real &z);

    /*! This method provides the inverse mapping from
     *  a point on the cosine-weighted hemisphere to a
     *  point on the unit square.
     *  \param p The point on the unit hemisphere.
     *  \param u The first coordinate of the corresponding
     *           point on the unit square is returned here.
     *  \param v The second coordinate of the corresponding
     *           point on the unit square is returned here.
     */
    inline static void inverse(const Real3 &p,
                               Real &u,
                               Real &v);

    /*! This method returns the size of the cosine-weighted
     *  hemisphere domain.
     *  \return 1.5 PI
     */
    inline static Real normalizationConstant(void);
}; // end UnitSquareToCosineHemisphere

#include "UnitSquareToCosineHemisphere.inl"

#endif // UNIT_SQUARE_TO_COSINE_HEMISPHERE_H

