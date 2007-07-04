/*! \file MaterialViewer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for MaterialViewer.h.
 */

#include "MaterialViewer.h"
#include "ScatteringFunction.h"
#include <printglerror/printGLError.h>

void MaterialViewer
  ::draw(void)
{
  Parent::draw();
  drawTexture(mTexture, mTexture2DRectProgram);
  printGLError(__FILE__, __LINE__);
} // end MaterialViewer::draw()

void MaterialViewer
  ::initTexture(void)
{
  float3 eye(0,0,1);

  Point o(-1.0, -1.0, 0);
  unsigned int w = 512, h = 512;
  float3 step(2.0f / w, 2.0f / h, 0);
  std::vector<float3> image(w*h);
  for(unsigned int y = 0; y < h; ++y)
  {
    for(unsigned int x = 0; x < w; ++x)
    {
      Point p = o + step * float3(x,y,0);
      float3 dpdu(1,0,0);
      float3 dpdv(0,1,0);
      float3 dndu(0,0,0);
      float3 dndv(0,0,0);
      ParametricCoordinates uv(float(x) / w, float(y) / h);

      DifferentialGeometry dg(p, dpdu, dpdv, dndu, dndv, uv, 0);
      ScatteringFunction *f = mMaterial->evaluate(dg);

      float3 wo = eye - dg.getPoint();
      wo = wo.normalize();
      float3 wi = mLightPosition - dg.getPoint();
      wi = wi.normalize();
      image[y*w + x] = 100.f * f->evaluate(wi,wo);

      // delete the scattering function
      delete f;
    } // end for x
  } // end for y

  // upload the image to the Texture
  mTexture.init(GL_FLOAT_RGB16_NV, w, h, 0, GL_FLOAT, (float*)&image[0]);
} // end MaterialViewer::initTexture()

void MaterialViewer
  ::setMaterial(Material *m)
{
  mMaterial = m;
} // end MaterialViewer::setMaterial()

void MaterialViewer
  ::init(void)
{
  Parent::init();
  mLightPosition = float3(0,0,1);
  mTexture.create();
  initTexture();
} // end MaterialViewer::init()

