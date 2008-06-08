/*! \file RasterizableMesh.inl
 *  \author Jared Hoberock
 *  \brief Inline file for RasterizableMesh.h.
 */

#include "RasterizableMesh.h"
#include <GL/glew.h>

template<typename MeshParentType>
  RasterizableMesh<MeshParentType>
    ::RasterizableMesh(const std::vector<Point> &vertices,
                       const std::vector<typename Parent0::Triangle> &triangles)
      :Parent0(vertices,triangles),Parent1()
{
  ;
} // end RasterizableMesh::RasterizableMesh()

template<typename MeshParentType>
  RasterizableMesh<MeshParentType>
    ::RasterizableMesh(const std::vector<Point> &points,
                       const std::vector<ParametricCoordinates> &parametrics,
                       const std::vector<typename Parent0::Triangle> &triangles)
      :Parent0(points,parametrics,triangles),Parent1()
{
  ;
} // end RasterizableMesh::RasterizableMesh()

template<typename MeshParentType>
  RasterizableMesh<MeshParentType>
    ::RasterizableMesh(const std::vector<Point> &points,
                       const std::vector<ParametricCoordinates> &parametrics,
                       const std::vector<Normal> &normals,
                       const std::vector<typename Parent0::Triangle> &triangles)
      :Parent0(points,parametrics,normals,triangles),Parent1()
{
  ;
} // end RasterizableMesh::RasterizableMesh()

template<typename MeshParentType>
  void RasterizableMesh<MeshParentType>
    ::rasterize(void)
{
  glPushAttrib(GL_POLYGON_BIT);
  glEnable(GL_CULL_FACE);
  glEnable(GL_NORMALIZE);

  const Mesh::TriangleList &triangles = Parent0::getTriangles();
  const Mesh::PointList &points = Parent0::getPoints();
  const Mesh::NormalList &normals = Parent0::getNormals();
  glBegin(GL_TRIANGLES);
  size_t triIndex = 0;
  for(Mesh::TriangleList::const_iterator t = triangles.begin();
      t != triangles.end();
      ++t, ++triIndex)
  {
    if(!Parent0::mInterpolateNormals)
    {
      // create a face normal
      glNormal3fv(normals[triIndex]);
    } // end if

    for(int i = 0; i < 3; ++i)
    {
      if(Parent0::mInterpolateNormals)
      {
        glNormal3fv(normals[(*t)[i]]);
      } // end if
      glVertex3fv(points[(*t)[i]]);
    } // end for v
  } // end for f
  glEnd();

  glPopAttrib();
} // end RasterizableMesh::rasterize()

