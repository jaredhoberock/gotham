/*! \file TriangleBVH.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of TriangleBVH class.
 */

#include "TriangleBVH.h"
#include "../surfaces/Mesh.h"
#include "SurfacePrimitive.h"

void TriangleBVH
  ::push_back(const boost::shared_ptr< ::Primitive> &p)
{
  // check that p is triangulatable
  // XXX DESIGN this sucks
  const SurfacePrimitive *sp = dynamic_cast<const SurfacePrimitive*>(p.get());
  if(sp)
  {
    const Mesh *mesh = dynamic_cast<const Mesh *>(sp->getSurface());
    if(mesh)
    {
      Parent0::push_back(p);
    } // end if
  } // end sp
} // end TriangleBVH::push_back()

void TriangleBVH
  ::finalize(void)
{
  // call the Parent first
  Parent0::finalize();

  // make a record of each primitive's triangles
  for(const_iterator prim = begin(); prim != end(); ++prim)
  {
    // XXX DESIGN this sucks
    const SurfacePrimitive *sp = dynamic_cast<const SurfacePrimitive*>(prim->get());
    const Mesh *mesh = dynamic_cast<const Mesh *>(sp->getSurface());

    if(mesh)
    {
      // make a record of the last primitive's triangles
      for(size_t i = 0; i != mesh->getTriangles().size(); ++i)
      {
        Triangle t;
        t.mPrimitive = sp;
        t.mTriangleIndex = i;
        mTriangles.push_back(t);
      } // end for i
    } // end if
  } // end for prim

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
  size_t triIndex = mBVH.mTriangles[tri].mTriangleIndex;
  const SurfacePrimitive *sp = mBVH.mTriangles[tri].mPrimitive;
  const Mesh *mesh = static_cast<const Mesh *>(sp->getSurface());

  // look up the Triangle
  Mesh::Triangle vertexIndices = mesh->getTriangles()[triIndex];
  return mesh->getPoints()[vertexIndices[vertexIndex]];
} // end TriangleVertexAccess::operator()()

void TriangleBVH
  ::getIntersection(const Point &p,
                    const float b1, const float b2,
                    const size_t triIndex,
                    Intersection &inter) const
{
  // figure out which triangle of which mesh we are
  // interested in
  const Triangle &globalTri = mTriangles[triIndex];
  size_t localTriIndex = globalTri.mTriangleIndex;
  const SurfacePrimitive *sp = globalTri.mPrimitive;

  // set the primitive in the intersection
  inter.setPrimitive(sp->getPrimitiveHandle());

  const Mesh *mesh = static_cast<const Mesh *>(sp->getSurface());
  const Mesh::PointList &points = mesh->getPoints();
  const Mesh::Triangle &tri = mesh->getTriangles()[localTriIndex];

  // create the DifferentialGeometry
  mesh->getDifferentialGeometry(tri, p,
                                b1,
                                b2,
                                inter.getDifferentialGeometry());
} // end TriangleBVH::getIntersection()

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
    // create an Intersection object
    getIntersection(r(t), b1, b2, triIndex, inter);

    // set the end of the Ray's interval
    // XXX DESIGN: Why do we even do this?
    r.getInterval()[1] = t;
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

