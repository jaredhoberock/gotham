/*! \file TriangleMesh.inl
 *  \author Jared Hoberock
 *  \brief Inline file for TriangleMesh.h.
 */

#include "TriangleMesh.h"

template<typename P3D, typename P2D, typename N3D>
  TriangleMesh<P3D, P2D, N3D>
    ::TriangleMesh(void)
{
  ;
} // end TriangleMesh::TriangleMesh()

template<typename P3D, typename P2D, typename N3D>
  TriangleMesh<P3D, P2D, N3D>
    ::TriangleMesh(const std::vector<P3D> &points,
                   const std::vector<Triangle> &triangles)
{
  mPoints = points;
  mTriangles = triangles;
} // end TriangleMesh::TriangleMesh()

template<typename P3D, typename P2D, typename N3D>
  TriangleMesh<P3D, P2D, N3D>
    ::TriangleMesh(const std::vector<P3D> &points,
                   const std::vector<P2D> &parametrics,
                   const std::vector<Triangle> &triangles)
{
  mPoints = points;
  mParametrics = parametrics;
  mTriangles = triangles;
} // end TriangleMesh::TriangleMesh()

template<typename P3D, typename P2D, typename N3D>
  const typename TriangleMesh<P3D, P2D, N3D>::TriangleList
    &TriangleMesh<P3D, P2D, N3D>
      ::getTriangles(void) const
{
  return mTriangles;
} // end TriangleMesh::getTriangles()

template<typename P3D, typename P2D, typename N3D>
  const typename TriangleMesh<P3D, P2D, N3D>::PointList
    &TriangleMesh<P3D, P2D, N3D>
      ::getPoints(void) const
{
  return mPoints;
} // end TriangleMesh::getPoints()

template<typename P3D, typename P2D, typename N3D>
  const typename TriangleMesh<P3D, P2D, N3D>::NormalList
    &TriangleMesh<P3D, P2D, N3D>
      ::getNormals(void) const
{
  return mNormals;
} // end TriangleMesh::getNormals()

template<typename P3D, typename P2D, typename N3D>
  float TriangleMesh<P3D, P2D, N3D>
    ::computeSurfaceArea(const Triangle &t) const
{
  P3D e0 = mPoints[t[1]] - mPoints[t[0]];
  P3D e1 = mPoints[t[2]] - mPoints[t[0]];
  return 0.5f * e0.cross(e1).norm();
} // end TriangleMesh::computeSurfaceArea()

