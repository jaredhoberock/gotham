/*! \file TriangleBVH.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a bounding volume hierarchy of triangles.
 */

#ifndef TRIANGLE_BVH_H
#define TRIANGLE_BVH_H

#include "TriangleList.h"
#include <boundingvolumehierarchy/TriangleBoundingVolumeHierarchy.h>
#include "../surfaces/Mesh.h"

class TriangleBVH
  : public TriangleList,
    protected TriangleBoundingVolumeHierarchy<size_t, gpcpu::float3, float>
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef TriangleList Parent0;

    /*! This method adds a new SurfacePrimitive to this
     *  SurfacePrimitiveList.
     *  \param p The SurfacePrimitive to add.
     *  \note If p's Surface is not triangulatable, it is not added to this
     *        TriangleList.
     */
    virtual void push_back(boost::shared_ptr<ListElement> &p);

    /*! This method finalizes this TriangleBVH.
     */
    virtual void finalize(void);

    /*! This method intersects a Ray against this TriangleBVH.
     *  \param r The Ray of interest.
     *  \param inter If an intersection exists, information regarding the
     *         geometry of the Intersection is returned here.
     *  \return true if an Intersection exists; false, otherwise.
     */
    virtual bool intersect(Ray &r, Intersection &inter) const;

    /*! This method intersects a shadow Ray against this TriangleBVH.
     *  \param r The Ray of interest.
     *  \return true if an intersection exists; false, otherwise.
     */
    virtual bool intersect(const Ray &r) const;

  protected:
    /*! \typedef Parent1
     *  \brief Shorthand.
     */
    typedef TriangleBoundingVolumeHierarchy<size_t, gpcpu::float3, float> Parent1;

    struct Triangle
    {
      size_t mPrimitiveIndex;
      size_t mTriangleIndex;
    }; // end Triangle

    std::vector<Triangle> mTriangles;

    struct TriangleVertexAccess
    {
      const TriangleBVH &mBVH;
      const gpcpu::float3 &operator()(const size_t tri,
                                      const size_t vertexIndex) const;
    }; // end TriangleVertexAccess
}; // end TriangleBVH

#endif // TRIANGLE_BVH_H

