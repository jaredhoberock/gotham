/*! \file RasterizableSphere.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RasterizableSphere class.
 */

#include "RasterizableSphere.h"
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

  gluSphere(q, mRadius, 20, 20);

  glPopMatrix();
  glPopAttrib();

  gluDeleteQuadric(q);
} // end RasterizableSphere::rasterize()

