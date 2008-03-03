/*! \file TriangleBVH.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of TriangleBVH class.
 */

#include "TriangleBVH.h"
#include "../surfaces/Mesh.h"
#include "SurfacePrimitive.h"

void TriangleBVH
  ::push_back(boost::shared_ptr<ListElement> &p)
{
  Parent0::push_back(p);

  // XXX DESIGN this sucks
  const SurfacePrimitive *sp = dynamic_cast<const SurfacePrimitive*>(p.get());
  if(sp)
  {
    const Mesh *mesh = dynamic_cast<const Mesh *>(sp->getSurface());
    if(mesh)
    {
      // make a record of the last primitive's triangles
      size_t primIndex = size() - 1;
      for(size_t i = 0; i != mesh->getTriangles().size(); ++i)
      {
        Triangle t;
        t.mPrimitiveIndex = primIndex;
        t.mTriangleIndex = i;
        mTriangles.push_back(t);
      } // end for i
    } // end if
  } // end if
} // end TriangleBVH::push_back()

void TriangleBVH
  ::finalize(void)
{
  // call the Parent first
  Parent0::finalize();

  // build a temporary array of triangle indices
  std::vector<size_t> tempTriangles(mTriangles.size());
  for(size_t i = 0; i != tempTriangles.size(); ++i)
  {
    tempTriangles[i] = i;
  } // end for i

  // build the BVH
  TriangleVertexAccess vertex = {*this};
  Parent1::build(tempTriangles, vertex);
} // end TriangleBVH::finalize()

const gpcpu::float3 &TriangleBVH::TriangleVertexAccess
  ::operator()(const size_t tri,
               const size_t vertexIndex) const
{
  // look up the primitive
  size_t prim = mBVH.mTriangles[tri].mPrimitiveIndex;
  size_t triIndex = mBVH.mTriangles[tri].mTriangleIndex;
  const SurfacePrimitive *sp = static_cast<const SurfacePrimitive*>(mBVH[prim].get());
  const Mesh *mesh = static_cast<const Mesh *>(sp->getSurface());

  // look up the Triangle
  Mesh::Triangle vertexIndices = mesh->getTriangles()[triIndex];
  return mesh->getPoints()[vertexIndices[vertexIndex]];
} // end TriangleVertexAccess::operator()()

bool TriangleBVH
  ::intersect(Ray &r, Intersection &inter) const
{
  float t, b1, b2;
  size_t triIndex;
  if(Parent1::intersect(r.getAnchor(),
                        r.getDirection(),
                        r.getInterval()[0], r.getInterval()[1],
                        t, b1, b2, triIndex))
  {
    // evaluate the differential geometry
    const Triangle &globalTri = mTriangles[triIndex];
    size_t meshIndex     = globalTri.mPrimitiveIndex;
    size_t localTriIndex = globalTri.mTriangleIndex;

    const SurfacePrimitive *sp = static_cast<const SurfacePrimitive*>((*this)[meshIndex].get());

    // set the primitive in the intersection
    inter.setPrimitive(sp);

    const Mesh *mesh = static_cast<const Mesh *>(sp->getSurface());
    const Mesh::PointList &points = mesh->getPoints();
    const Mesh::Triangle &tri = mesh->getTriangles()[localTriIndex];

    // XXX the geometric normal should be contained within the Wald-Bikker data
    Vector3 e1 = points[tri[1]] - points[tri[0]];
    Vector3 e2 = points[tri[2]] - points[tri[0]];

    // set the end of the Ray's interval
    // XXX DESIGN: Why do we even do this?
    r.getInterval()[1] = t;
    mesh->getDifferentialGeometry(tri, r(t),
                                  e1.cross(e2).normalize(),
                                  b1,
                                  b2,
                                  inter.getDifferentialGeometry());
    inter.getDifferentialGeometry().setSurface(static_cast<const SurfacePrimitive*>(inter.getPrimitive()));
    return true;
  } // end if

  return false;
} // end TriangleBVH::intersect()

bool TriangleBVH
  ::intersect(const Ray &r) const
{
  return Parent1::shadow(r.getAnchor(),
                         r.getDirection(),
                         r.getInterval()[0],
                         r.getInterval()[1]);
} // end TriangleBVH::intersect()

