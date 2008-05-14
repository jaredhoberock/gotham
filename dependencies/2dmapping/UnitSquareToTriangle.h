/*! \file UnitSquareToTriangle.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class for mapping
 *         from the unit square to a triangle.
 */

#ifndef UNIT_SQUARE_TO_TRIANGLE_H
#define UNIT_SQUARE_TO_TRIANGLE_H

class UnitSquareToTriangle
{
  public:
    /*! This method maps points in the unit square to
     *  points uniformly distributed over a triangle
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param v0 The first vertex of the triangle.
     *  \param v1 The second vertex of the triangle.
     *  \param v2 The third vertex of the triangle.
     *  \param p The mapping (u,v) -> to a point in the given triangle
     *           is returned here.
     *  \param pdf The value of the distribution function over the triangle
     *             is returned here. This is simply the inverse of the triangle's
     *             area.
     */
    template<typename Real, typename Point>
      static void evaluate(const Real &u, const Real &v,
                           const Point &v0, const Point &v1, const Point &v2,
                           Point &p,
                           Real *pdf = 0);

    /*! This method maps poitns in the unit square to
     *  points uniformly distributed over a triangle.
     *  \param u The first coordinate of the point in the unit square.
     *  \param v The second coordinate of the point in the unit square.
     *  \param v0 The first vertex of the triangle.
     *  \param v1 The second vertex of the triangle.
     *  \param v2 The third vertex of the triangle.
     *  \param p The mapping (u,v) -> to a point in the given triangle is
     *           returned here.
     *  \param b The first barycentric coordinates of p are returned here.
     *  \param pdf The value of the distribution function over the triangle
     *             is returned here. This is simply the inverse of the triangle's
     *             area.
     */
    template<typename Real, typename Point, typename Point2>
      static void evaluate(const Real &u, const Real &v,
                           const Point &v0, const Point &v1, const Point &v2,
                           Point &p, Point2 &b,
                           Real *pdf = 0);

    /*! This method provides the inverse mapping from a point on the triangle
     *  to a point on the unit square.
     *  \param p The point on the triangle.
     *  \param v0 The first vertex of the triangle.
     *  \param v1 The second vertex of the triangle.
     *  \param v2 The third vertex of the triangle.
     *  \param u The first coordinate of the corresponding
     *           point on the unit square is returned here.
     *  \param v The second coordinate of the corresponding
     *           point on the unit square is returned here.
     */
    template<typename Point, typename Real>
      static void inverse(const Point &p,
                          const Point &v0,
                          const Point &v1,
                          const Point &v2,
                          Real &u,
                          Real &v);

    /*! This method evaluates the probability density
     *  function of this mapping at the given point
     *  on the triangle.
     *  \param p The point of interest on the triangle.
     *  \param v0 The first vertex of the triangle.
     *  \param v1 The second vertex of the triangle.
     *  \param v2 The third vertex of the triangle.
     *  \return The value of the pdf at p.  This is the inverse of the triangle's area.
     */
    template<typename Real, typename Point>
      static Real evaluatePdf(const Point &p, const Point &v0, const Point &v1, const Point &v2);
}; // end UnitSquareToTriangle

#include "UnitSquareToTriangle.inl"

#endif // UNIT_SQUARE_TO_TRIANGLE_H

