/*! \file RasterizableSphere.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RasterizableSphere class.
 */

#include "RasterizableSphere.h"
#include "../geometry/Mappings.h"
#include <GL/glew.h>

RasterizableSphere
  ::RasterizableSphere(const Point &c,
                       const float r)
    :Parent0(c,r),Parent1()
{
} // end RasterizableSphere::RasterizableSphere()

void RasterizableSphere
  ::rasterize(void)
{
  GLUquadric *q = gluNewQuadric();

  glPushAttrib(GL_POLYGON_BIT);
  glEnable(GL_CULL_FACE);

  // push the transform
  glPushMatrix();

  glTranslatef(mCenter[0], mCenter[1], mCenter[2]);
  
  size_t uDivisions = 20;
  size_t vDivisions = 20;
  float uDel = 1.0f / uDivisions;
  float vDel = 1.0f / vDivisions;

  glScalef(mRadius, mRadius, mRadius);

  glBegin(GL_QUADS);
  float v = 0;
  for(size_t j = 0; j != uDivisions; ++j, v += vDel)
  {
    float u = 0;
    for(size_t i = 0; i != 20; ++i, u += uDel)
    {
      Point p;
      float pdf;
      Mappings<Point>::unitSquareToSphere(u, v,
                                          Point(1,0,0),
                                          Point(0,1,0),
                                          Point(0,0,1),
                                          p, pdf);

      glNormal3fv(p);
      glVertex3fv(p);

      Mappings<Point>::unitSquareToSphere(u + uDel, v,
                                          Point(1,0,0),
                                          Point(0,1,0),
                                          Point(0,0,1),
                                          p, pdf);
      glNormal3fv(p);
      glVertex3fv(p);

      Mappings<Point>::unitSquareToSphere(u + uDel, v + vDel,
                                          Point(1,0,0),
                                          Point(0,1,0),
                                          Point(0,0,1),
                                          p, pdf);
      glNormal3fv(p);
      glVertex3fv(p);

      Mappings<Point>::unitSquareToSphere(u, v + vDel,
                                          Point(1,0,0),
                                          Point(0,1,0),
                                          Point(0,0,1),
                                          p, pdf);
      glNormal3fv(p);
      glVertex3fv(p);
    } // end for
  } // end for j
  glEnd();

  glPopMatrix();
  glPopAttrib();

  gluDeleteQuadric(q);
} // end RasterizableSphere::rasterize()

