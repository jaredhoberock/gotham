/*! \file SmallMesh.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SmallMesh class.
 */

#include "SmallMesh.h"

SmallMesh
  ::SmallMesh(const std::vector<Point> &vertices,
              const std::vector<Triangle> &triangles)
    :Parent(vertices,triangles)
{
  ;
} // end SmallMesh::SmallMesh()

SmallMesh
  ::SmallMesh(const std::vector<Point> &vertices,
              const std::vector<ParametricCoordinates> &parametrics,
              const std::vector<Triangle> &triangles)
    :Parent(vertices,parametrics,triangles)
{
  ;
} // end SmallMesh::SmallMesh()

bool SmallMesh
  ::intersect(const Ray &r) const
{
  float t, b1, b2;
  TriangleList::const_iterator end = mTriangles.end();
  for(TriangleList::const_iterator tri = mTriangles.begin();
      tri != end;
      ++tri)
  {
    if(Parent::intersectWaldBikker(r.getAnchor(), r.getDirection(), *tri, *this, r.getInterval()[0], r.getInterval()[1], t, b1, b2))
    {
      return true;
    } // end if
  } // end for t

  return false;
} // end SmallMesh::intersect()

bool SmallMesh
  ::intersect(const Ray &r, float &t, DifferentialGeometry &dg) const
{
  bool result = false;
  float tempt, tempb1, tempb2;
  float b1, b2;
  const Triangle *hitTri = 0;
  TriangleList::const_iterator end = mTriangles.end();
  for(TriangleList::const_iterator tri = mTriangles.begin();
      tri != end;
      ++tri)
  {
    if(Parent::intersectWaldBikker(r.getAnchor(), r.getDirection(), *tri, *this, r.getInterval()[0], r.getInterval()[1], tempt, tempb1, tempb2))
    {
      t = tempt;
      b1 = tempb1;
      b2 = tempb2;
      result = true;
      hitTri = &*(tri);
    } // end if
  } // end for t

  if(result)
  {
    // fill out DifferentialGeometry details
    Vector3 e1 = mPoints[(*hitTri)[1]] - mPoints[(*hitTri)[0]];
    Vector3 e2 = mPoints[(*hitTri)[2]] - mPoints[(*hitTri)[0]];
    getDifferentialGeometry(*hitTri, r(t), e1.cross(e2).normalize(),
                            b1,
                            b2,
                            dg);
  } // end if

  return result;
} // end SmallMesh::intersect()

