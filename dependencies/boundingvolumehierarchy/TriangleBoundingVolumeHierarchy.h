/*! \file TriangleBoundingVolumeHieararchy.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class representing
 *         a hierarchy of axis aligned bounding boxes
 *         over a set of triangles.
 */

#ifndef TRIANGLE_BOUNDING_VOLUME_HIERARCHY_H
#define TRIANGLE_BOUNDING_VOLUME_HIERARCHY_H

#include "BoundingVolumeHierarchy.h"

template<typename TriangleType, typename PointType, typename RealType = float>
  class TriangleBoundingVolumeHierarchy
    : protected BoundingVolumeHierarchy<TriangleType,PointType,RealType>
{
  public:
    /*! \typedef Triangle
     *  \brief Shorthand.
     */
    typedef TriangleType Triangle;

    /*! This method builds this TriangleBoundingVolumeHierarchy.
     *  \param triangles A list of Triangles.
     *  \param vertex A VertexAccess functor. vertex(tri,i) provides
     *                access to the ith PointType of Triangle tri.
     */
    template<typename VertexAccess>
      void build(const std::vector<Triangle> &triangles,
                 VertexAccess &vertex);

    /*! This method intersects a ray against this TriangleBoundingVolumeHierarchy.
     *  \param o The origin of the ray of interest.
     *  \param d The direction of the ray of interest.
     *  \param tMin The minimum ray parameter to consider.
     *  \param tMax The maximum ray parameter to consider.
     *  \param t If an intersection exists, the hit time is returned here.
     *  \param b1 If an intersection exists, the first barycentric coordinate at the hit point is returned here.
     *  \param b2 If an intersection exists, the second barycentric coordinate at the hit point is returned here.
     *  \param tri If an intersection exists, the index of the primitive is returned here.
     */
    bool intersect(const PointType &o, const PointType &d,
                   const RealType &tMin, RealType tMax,
                   RealType &t, RealType &b1, RealType &b2, size_t &tri) const;

    /*! This method intersects a shadow ray against this TriangleBoundingVolumeHierarchy.
     *  \param o The origin of the ray of interest.
     *  \param d The direction of the ray of interest.
     *  \param tMin The minimum ray parameter to consider.
     *  \param tMax The maximum ray parameter to consider.
     *  \return true if the ray hits something; false, otherwise.
     */
    bool shadow(const PointType &o, const PointType &d,
                const RealType &tMin, const RealType &tMax) const;


  protected:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef BoundingVolumeHierarchy<TriangleType,PointType,RealType> Parent;

    template<typename VertexAccess>
      struct TriangleBounder
      {
        inline RealType operator()(const size_t axis,
                                   const bool minimum,
                                   const Triangle &tri) const;
        VertexAccess &mVertex;
      }; // end TriangleBounder

    /*! This method builds Wald-Bikker ray-triangle intersection
     *  data for each Triangle.
     */
    template<typename VertexAccess>
      void buildWaldBikkerData(const std::vector<Triangle> &triangles,
                               VertexAccess &vertex);

    /*! Additional per-triangle data which does not fit inside
     *  Parent::mMinBoundsHitIndex & Parent::mMaxBoundsHitIndex
     *  This holds the first vertex of the triangle and also
     *  the index of the dominant axis.
     */
    std::vector<gpcpu::float4> mFirstVertexDominantAxis;
}; // end TriangleBoundingVolumeHierarchy

#include "TriangleBoundingVolumeHierarchy.inl"

#endif // TRIANGLE_BOUNDING_VOLUME_HIERARCHY_H

