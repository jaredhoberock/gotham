/*! \file waldBikkerIntersection.inl
 *  \author Jared Hoberock
 *  \brief Inline file for waldBikkerIntersection.h.
 */

#include "waldBikkerIntersection.h"
#include <math.h>

template<typename PointType, typename RealType>
  void buildWaldBikkerIntersectionData(const PointType &v0,
                                       const PointType &v1,
                                       const PointType &v2,
                                       PointType &n,
                                       unsigned int &dominantAxis,
                                       RealType &bnu, RealType &bnv,
                                       RealType &cnu, RealType &cnv)
{
  PointType b = v2 - v0;
  PointType c = v1 - v0;
  PointType normal = c.cross(b).normalize();

  // determine dominant axis
  if(fabs(normal[0]) > fabs(normal[1]))
  {
    if(fabs(normal[0]) > fabs(normal[2])) dominantAxis = 0;
    else dominantAxis = 2;
  } // end if
  else
  {
    if(fabs(normal[1]) > fabs(normal[2])) dominantAxis = 1;
    else dominantAxis = 2;
  } // end else

  size_t u = (dominantAxis + 1) % 3;
  size_t v = (dominantAxis + 2) % 3;

  n[0] = normal.dot(v0) / normal[dominantAxis];
  n[1] = normal[u] / normal[dominantAxis];
  n[2] = normal[v] / normal[dominantAxis];

  bnu =  b[u] / (b[u] * c[v] - b[v] * c[u]);
  bnv = -b[v] / (b[u] * c[v] - b[v] * c[u]);

  cnu =  c[v] / (b[u] * c[v] - b[v] * c[u]);
  cnv = -c[u] / (b[u] * c[v] - b[v] * c[u]);
} // end buildWaldBikkerIntersectionData()

template<typename PointType, typename RealType>
  bool waldBikkerIntersection(const PointType &o,
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
                              RealType &b2)
{
  size_t uAxis = (dominantAxis + 1) % 3;
  size_t vAxis = (dominantAxis + 2) % 3;

  // calculate distance to triangle plane
  float denom = (     dir[dominantAxis] + n[1] * dir[uAxis] + n[2] * dir[vAxis]);
  float numer = (n[0] - o[dominantAxis] - n[1] *   o[uAxis] - n[2] *   o[vAxis]);
  t = numer / denom;

  // if the triangle is malformed (e.g., zero area)
  // then, at this point, t is nan
  // write this comparison so that t must be non-nan
  // to continue
  if(t >= minT && t <= maxT)
  {
    // calculate hit point
    float pu = o[uAxis] + t * dir[uAxis] - v0[uAxis];
    float pv = o[vAxis] + t * dir[vAxis] - v0[vAxis];
    b1 = pv * bnu + pu * bnv;
    if(b1 < 0) return false;
    b2 = pu * cnu + pv * cnv;
    if(b2 < 0) return false;

    if(b1 + b2 > 1.0f) return false;

    return true;
  } // end if

  return false;
} // end waldBikkerIntersection()

