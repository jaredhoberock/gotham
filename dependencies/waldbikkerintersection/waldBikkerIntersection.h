/*! \file waldBikkerIntersection.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to functions implementing
 *         Wald-Bikker fast ray-triangle intersection.
 */

#ifndef WALD_BIKKER_INTERSECTION_H
#define WALD_BIKKER_INTERSECTION_H

/*! This function builds Wald-Bikker fast ray-triangle intersection
 *  data for a triangle.
 *  \param v0 The first vertex of the triangle.
 *  \param v1 The second vertex of the triangle.
 *  \param v2 The third vertex of the triangle.
 *  \param n The N' data is returned here.
 *  \param dominantAxis The index of the dominant axis of the triangle
 *                      normal is returned here.
 *  \param bnu More data.
 *  \param bnu More data.
 *  \param cnu More data.
 *  \param cnv More Data.
 */
template<typename PointType, typename RealType>
  inline void buildWaldBikkerIntersectionData(const PointType &v0,
                                              const PointType &v1,
                                              const PointType &v2,
                                              PointType &n,
                                              unsigned int &dominantAxis,
                                              RealType &bnu, RealType &bnv,
                                              RealType &cnu, RealType &cnv);

/*! This function intersects a ray with a triangle stored in the fast Wald-Bikker
 *  ray-triangle intersection format.
 *  \param o The origin of the ray of interest.
 *  \param dir The direction of the ray of interest.
 *  \param minT The minimum ray parameter to consider for intersection.
 *  \param maxT The maximum ray parameter to consider for intersection.
 *  \param v0 The first vertex of the triangle of interest.
 *  \param n The 'n' result returned by buildWaldBikkerIntersectionData() for this triangle.
 *  \param dominantAxis The 'dominantAxis' result returned by buildWaldBikkerIntersectionData() for this triangle.
 *  \param bnu The 'bnu' result returned by buildWaldBikkerIntersectionData() for this triangle.
 *  \param bnv The 'bnv' result returned by buildWaldBikkerIntersectionData() for this triangle.
 *  \param cnu The 'cnu' result returned by buildWaldBikkerIntersectionData() for this triangle.
 *  \param cnv The 'cnv' result returned by buildWaldBikkerIntersectionData() for this triangle.
 *  \param t if an intersection exists, the hit time is returned here; otherwise, it is undefined.
 *  \param b1 if an intersection exists, the first triangle barycentric coordinate at the hit point is returned here;
 *            otherwise, the result returned in this parameter is undefined.
 *  \param b2 if an intersection exists, the second triangle barycentric coordinate at the hit point is returned here;
 *            otherwise, the result returned in this parameter is undefined.
 *  \note [minT,maxT] defines a closed interval.
 */
template<typename PointType, typename RealType>
  inline bool waldBikkerIntersection(const PointType &org,
                                     const PointType &dir,
                                     const RealType minT,
                                     const RealType maxT,
                                     const PointType &v0,
                                     const PointType &n,
                                     const unsigned int &dominantAxis,
                                     const RealType &bnu, const RealType &bnv,
                                     const RealType &cnu, const RealType &cnv,
                                     RealType &t,
                                     RealType &b1,
                                     RealType &b2);

#include "waldBikkerIntersection.inl"

#endif // WALD_BIKKER_INTERSECTION_H

