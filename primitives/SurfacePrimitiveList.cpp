/*! \file SurfacePrimitiveList.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SurfacePrimitiveList class.
 */

#include "SurfacePrimitiveList.h"

SurfacePrimitiveList
  ::SurfacePrimitiveList(void)
    :Parent()
{
  mSurfaceArea = 0;
} // end SurfacePrimitiveList::SurfacePrimitiveList()

void SurfacePrimitiveList
  ::sampleSurfaceArea(const float u,
                      const SurfacePrimitive **prim,
                      float &pdf) const
{
  *prim = mSurfaceAreaPdf(u, pdf);
} // end SurfacePrimitive::sampleSurfaceArea()

void SurfacePrimitiveList
  ::push_back(boost::shared_ptr<SurfacePrimitive> &p)
{
  Parent::push_back(p);
  mSurfaceArea += p->getSurfaceArea();
  mOneOverSurfaceArea = 1.0f / mSurfaceArea;
} // end SurfacePrimitive::push_back()

void SurfacePrimitiveList
  ::finalize(void)
{
  buildSurfaceAreaPdf();
} // end SurfacePrimitive::finalize()

void SurfacePrimitiveList
  ::buildSurfaceAreaPdf(void)
{
  std::vector<const SurfacePrimitive*> pointers;
  std::vector<float> area;
  for(const_iterator p = begin();
      p != end();
      ++p)
  {
    pointers.push_back((*p).get());
    area.push_back((*p)->getSurfaceArea());
  } // end for t

  mSurfaceAreaPdf.build(pointers.begin(), pointers.end(),
                        area.begin(), area.end());
} // end SurfacePrimitiveList::buildSurfaceAreaPdf()

void SurfacePrimitiveList
  ::sampleSurfaceArea(const float u0,
                      const float u1,
                      const float u2,
                      const float u3,
                      const SurfacePrimitive **surf,
                      DifferentialGeometry &dg,
                      float &pdf) const
{
  // sample a Surface
  sampleSurfaceArea(u0, surf, pdf);

  // sample from surf
  float temp;
  (*surf)->sampleSurfaceArea(u1,u2,u3,dg,temp);
  pdf *= temp;
} // end SurfacePrimitiveList::sampleSurfaceArea()

