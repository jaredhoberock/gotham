/*! \file UnitSquareToIsoscelesRightTriangle.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class for mapping
 *         from the unit square to the unit isosceles
 *         right triangle of area 1/2.
 */

#ifndef UNIT_SQUARE_TO_ISOSCELES_RIGHT_TRIANGLE_H
#define UNIT_SQUARE_TO_ISOSCELES_RIGHT_TRIANGLE_H

class UnitSquareToIsoscelesRightTriangle
{
  public:
    /*! This method maps points in the unit square to
     *  points uniformly distributed over the unit
     *  isosceles right triangle.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param p The mapping (u,v) -> to a point in the unit isosceles
     *           right triangle is returned here.
     *  \param pdf The value of the distribution function over the unit
     *             isosceles right triangle is returned here.
     */
    template<typename Real, typename Real2>
      static void evaluate(const Real &u,
                           const Real &v,
                           Real2 &p,
                           Real *pdf = 0);

    /*! This method provides the inverse mapping from
     *  a point on the unit isosceles right triangle to a point
     *  on the unit square.
     *  \param p The point on the unit isosceles right triangle.
     *  \param u The first coordinate of the corresponding
     *           point on the unit square is returned here.
     *  \param v The second coordinate of the corresponding
     *           point on the unit square is returned here.
     */
    template<typename Real2, typename Real>
      static void inverse(const Real2 &p,
                          Real &u,
                          Real &v);

    /*! This method evaluates the probability density
     *  function of this mapping at the given point
     *  on the unit isosceles right triangle.
     *  \param p The point of interest in the unit
     *         isosceles right triangle.
     *  \return The value of the pdf at p.
     */
    template<typename Real, typename Real2>
      static Real evaluatePdf(const Real2 &p);
}; // end UnitSquareToIsoscelesRightTriangle

#include "UnitSquareToIsoscelesRightTriangle.inl"

#endif // UNIT_SQUARE_TO_ISOSCELES_RIGHT_TRIANGLE_H

