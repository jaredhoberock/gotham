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
  for(Mesh::TriangleList::const_iterator t = triangles.begin();
      t != triangles.end();
      ++t)
  {
    if(normals.empty())
    {
      // create a face normal
      Vector e1 = points[(*t)[1]] - points[(*t)[0]];
      Vector e2 = points[(*t)[2]] - points[(*t)[0]];

      Normal n = e1.cross(e2);
      n = n.normalize();
      glNormal3fv(n);
    } // end if

    for(int i = 0; i < 3; ++i)
    {
      if(!normals.empty())
      {
        glNormal3fv(normals[(*t)[i]]);
      } // end if
      glVertex3fv(points[(*t)[i]]);
    } // end for v
  } // end for f
  glEnd();

  glPopAttrib();
} // end RasterizableMesh::rasterize()

