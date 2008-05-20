/*! \file TriangleBoundingVolumeHierarchy.inl
 *  \author Jared Hoberock
 *  \brief Inline file for TriangleBoundingVolumeHierarchy.h.
 */

#include "TriangleBoundingVolumeHierarchy.h"
#include <bittricks/bittricks.h>
#include <gpcpu/Vector.h>
#include <waldbikkerintersection/waldBikkerIntersection.h>

template<typename TriangleType, typename PointType, typename RealType>
  template<typename VertexAccess>
    void TriangleBoundingVolumeHierarchy<TriangleType,PointType,RealType>
      ::build(const std::vector<Triangle> &triangles,
              VertexAccess &vertex)
{
  // create a bounder
  TriangleBounder<VertexAccess> bounder = {vertex};
  Parent::build(triangles, bounder);

  // create Wald-Bikker data for each triangle
  buildWaldBikkerData(triangles, vertex);
} // end TriangleBoundingVolumeHierarchy::build()

template<typename TriangleType, typename PointType, typename RealType>
  template<typename VertexAccess>
    RealType TriangleBoundingVolumeHierarchy<TriangleType,PointType,RealType>
      ::TriangleBounder<VertexAccess>
        ::operator()(const size_t axis,
                     const bool minimum,
                     const Triangle &tri) const
{
  RealType result = std::numeric_limits<RealType>::infinity();
  if(!minimum) result = -result;
  for(size_t i = 0;
      i != 3;
      ++i)
  {
    if(minimum)
    {
      result = std::min(result, mVertex(tri, i)[axis]);
    } // end if
    else
    {
      result = std::max(result, mVertex(tri, i)[axis]);
    } // end else
  } // end for i

  return result;
} // end TriangleBounder::operator()()

template<typename TriangleType, typename PointType, typename RealType>
  template<typename VertexAccess>
    void TriangleBoundingVolumeHierarchy<TriangleType,PointType,RealType>
      ::buildWaldBikkerData(const std::vector<Triangle> &triangles,
                            VertexAccess &vertex)
{
  mFirstVertexDominantAxis.clear();

  // we will stick much of the data in mMinBounds & mMaxBounds
  size_t i = 0;
  for(typename std::vector<Triangle>::const_iterator tri = triangles.begin();
      tri != triangles.end();
      ++tri, ++i)
  {
    Point v0 = vertex(*tri, 0);
    Point v1 = vertex(*tri, 1);
    Point v2 = vertex(*tri, 2);

    Point n;
    unsigned int dominantAxis;
    RealType bnu, bnv, cnu, cnv;

    buildWaldBikkerIntersectionData<PointType,RealType>(v0,v1,v2,
                                                        n, dominantAxis,
                                                        bnu, bnv,
                                                        cnu, cnv);

    // stick n in mMinBoundsHitIndex.xyz
    Parent::mMinBoundHitIndex[i][0] = n[0];
    Parent::mMinBoundHitIndex[i][1] = n[1];
    Parent::mMinBoundHitIndex[i][2] = n[2];

    // we can overwrite the miss index, because it is always
    // identical to the hit index for leaf nodes (triangles)

    // stick bnu, bnv, cnu, cnv inside mMaxBoundsHitIndex
    Parent::mMaxBoundMissIndex[i][0] = bnu;
    Parent::mMaxBoundMissIndex[i][1] = bnv;
    Parent::mMaxBoundMissIndex[i][2] = cnu;
    Parent::mMaxBoundMissIndex[i][3] = cnv;

    // add the extra information to the additional list
    gpcpu::float4 temp(v0[0], v0[1], v0[2], uintAsFloat(dominantAxis));
    mFirstVertexDominantAxis.push_back(temp);
  } // end for i
} // end TriangleBoundingVolumeHierarchy::buildWaldBikkerData()

template<typename TriangleType, typename PointType, typename RealType>
  bool TriangleBoundingVolumeHierarchy<TriangleType,PointType,RealType>
    ::intersect(const PointType &o, const PointType &d,
                const RealType &tMin, RealType tMax,
                RealType &t, RealType &b1, RealType &b2, size_t &tri) const
{
  PointType invDir;
  invDir[0] = RealType(1.0) / d[0];
  invDir[1] = RealType(1.0) / d[1];
  invDir[2] = RealType(1.0) / d[2];

  typename Parent::NodeIndex currentNode = Parent::mRootIndex;
  bool hit = false;
  bool result = false;
  t = tMax;
  gpcpu::float4 minBoundsHit, maxBoundsMiss;
  gpcpu::float4 firstVertexDominantAxis;

  // XXX PERF: it might be possible to eliminate these temporaries
  float tempT, tempB1, tempB2;
  while(currentNode != Parent::NULL_NODE)
  {
    minBoundsHit  = Parent::mMinBoundHitIndex[currentNode];
    maxBoundsMiss = Parent::mMaxBoundMissIndex[currentNode];

    // leaves (primitives) are listed before interior nodes
    // so bounding boxes occur after the root index
    if(currentNode >= Parent::mRootIndex)
    {
      hit = intersectBox(o, invDir,
                         Point(minBoundsHit[0], minBoundsHit[1], minBoundsHit[2]),
                         Point(maxBoundsMiss[0], maxBoundsMiss[1], maxBoundsMiss[2]),
                         tMin, tMax);
    } // end if
    else
    {
      firstVertexDominantAxis = mFirstVertexDominantAxis[currentNode];
      hit = waldBikkerIntersection<PointType,RealType>
        (o,d,tMin,tMax,
         Point(firstVertexDominantAxis[0],
               firstVertexDominantAxis[1],
               firstVertexDominantAxis[2]),
         Point(minBoundsHit[0],
               minBoundsHit[1],
               minBoundsHit[2]),
         floatAsUint(firstVertexDominantAxis[3]),
         maxBoundsMiss[0], maxBoundsMiss[1],
         maxBoundsMiss[2], maxBoundsMiss[3],
         tempT, tempB1, tempB2);
      result |= hit;

      // XXX we could potentially merge t and tMax into a single word
      //     as they serve essentially the same purpose
      if(hit)
      {
        t = tempT;
        tMax = tempT;
        tri = currentNode;
        b1 = tempB1;
        b2 = tempB2;
      } // end if

      // ensure that the miss and hit indices are the same
      // at this point
      maxBoundsMiss[3] = minBoundsHit[3];
    } // end else

    currentNode = hit ? floatAsUint(minBoundsHit[3]) : floatAsUint(maxBoundsMiss[3]);
  } // end while

  return result;
} // end TriangleBoundingVolumeHierarchy::intersect()

template<typename TriangleType, typename PointType, typename RealType>
  bool TriangleBoundingVolumeHierarchy<TriangleType,PointType,RealType>
    ::shadow(const PointType &o, const PointType &d,
             const RealType &tMin, const RealType &tMax) const
{
  PointType invDir;
  invDir[0] = RealType(1.0) / d[0];
  invDir[1] = RealType(1.0) / d[1];
  invDir[2] = RealType(1.0) / d[2];

  typename Parent::NodeIndex currentNode = Parent::mRootIndex;
  gpcpu::float4 minBoundsHit, maxBoundsMiss;
  gpcpu::float4 firstVertexDominantAxis;
  bool hit = false;

  // XXX PERF: it might be possible to eliminate these temporaries
  float tempT, tempB1, tempB2;
  while(currentNode != Parent::NULL_NODE)
  {
    minBoundsHit  = Parent::mMinBoundHitIndex[currentNode];
    maxBoundsMiss = Parent::mMaxBoundMissIndex[currentNode];

    // leaves (primitives) are listed before interior nodes
    // so bounding boxes occur after the root index
    if(currentNode >= Parent::mRootIndex)
    {
      hit = intersectBox(o, invDir,
                         Point(minBoundsHit[0], minBoundsHit[1], minBoundsHit[2]),
                         Point(maxBoundsMiss[0], maxBoundsMiss[1], maxBoundsMiss[2]),
                         tMin, tMax);
    } // end if
    else
    {
      firstVertexDominantAxis = mFirstVertexDominantAxis[currentNode];
      if(waldBikkerIntersection<PointType,RealType>
        (o,d,tMin,tMax,
         Point(firstVertexDominantAxis[0],
               firstVertexDominantAxis[1],
               firstVertexDominantAxis[2]),
         Point(minBoundsHit[0],
               minBoundsHit[1],
               minBoundsHit[2]),
         floatAsUint(firstVertexDominantAxis[3]),
         maxBoundsMiss[0], maxBoundsMiss[1],
         maxBoundsMiss[2], maxBoundsMiss[3],
         tempT, tempB1, tempB2))
      {
        return true;
      } // end if

      // ensure that the miss and hit indices are the same
      // at this point
      maxBoundsMiss[3] = minBoundsHit[3];
    } // end else

    currentNode = hit ? floatAsUint(minBoundsHit[3]) : floatAsUint(maxBoundsMiss[3]);
  } // end while

  return false;
} // end TriangleBoundingVolumeHierarchy::shadow()

