/*! \file MeshRasterizer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of MeshRasterizer class.
 */

#include "MeshRasterizer.h"

#ifdef WIN32
#include <windows.h>
#endif // WIN32

#include <GL/glew.h>
#include <GL/gl.h>

MeshRasterizer
  ::MeshRasterizer(boost::shared_ptr<Mesh> mesh)
    :Parent(mesh)
{
  ;
} // end MeshRasterizer::MeshRasterizer()

void MeshRasterizer
  ::operator()(void)
{
  glPushAttrib(GL_POLYGON_BIT);
  glEnable(GL_CULL_FACE);
  glEnable(GL_NORMALIZE);

  const Mesh::TriangleList &triangles = mPrimitive->getTriangles();
  const Mesh::PointList &points = mPrimitive->getPoints();
  const Mesh::NormalList &normals = mPrimitive->getNormals();
  glBegin(GL_TRIANGLES);
  for(Mesh::TriangleList::const_iterator t = triangles.begin();
      t != triangles.end();
      ++t)
  {
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
} // end MeshRasterizer::operator()()

