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

bool SurfacePrimitiveList
  ::sampleSurfaceArea(const float u,
                      const SurfacePrimitive **prim,
                      float &pdf) const
{
  if(!mSurfaceAreaPdf.empty())
  {
    *prim = mSurfaceAreaPdf(u, pdf);
    return true;
  } // end if

  return false;
} // end SurfacePrimitive::sampleSurfaceArea()

float SurfacePrimitiveList
  ::evaluateSurfaceAreaPdf(const SurfacePrimitive *prim) const
{
  return mSurfaceAreaPdf.evaluatePdf(prim);
} // end SurfacePrimitiveList::evaluateSurfaceAreaPdf()

void SurfacePrimitiveList
  ::push_back(const boost::shared_ptr<Primitive> &p)
{
  const SurfacePrimitive *sp = dynamic_cast<SurfacePrimitive*>(p.get());
  if(sp)
  {
    Parent::push_back(p);
    mSurfaceArea += sp->getSurfaceArea();
    mOneOverSurfaceArea = 1.0f / mSurfaceArea;
  } // end if
} // end SurfacePrimitive::push_back()

void SurfacePrimitiveList
  ::finalize(void)
{
  Parent::finalize();
  buildSurfaceAreaPdf();

  for(size_t i = 0; i != size(); ++i)
  {
    dynamic_cast<SurfacePrimitive*>((*this)[i].get())->setSurfacePrimitiveHandle(i);
  } // end for i
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
    const SurfacePrimitive *sp = dynamic_cast<const SurfacePrimitive*>((*p).get());
    pointers.push_back(sp);
    area.push_back(sp->getSurfaceArea());
  } // end for t

  mSurfaceAreaPdf.build(pointers.begin(), pointers.end(),
                        area.begin(), area.end());
} // end SurfacePrimitiveList::buildSurfaceAreaPdf()

bool SurfacePrimitiveList
  ::sampleSurfaceArea(const float u0,
                      const float u1,
                      const float u2,
                      const float u3,
                      const SurfacePrimitive **surf,
                      DifferentialGeometry &dg,
                      float &pdf) const
{
  // sample a Surface
  if(sampleSurfaceArea(u0, surf, pdf))
  {
    // sample from surf
    float temp;
    (*surf)->sampleSurfaceArea(u1,u2,u3,dg,temp);
    pdf *= temp;
    return true;
  } // end if

  return false;
} // end SurfacePrimitiveList::sampleSurfaceArea()

float SurfacePrimitiveList
  ::evaluateSurfaceAreaPdf(const SurfacePrimitive *surf,
                           const DifferentialGeometry &dg) const
{
  float result = evaluateSurfaceAreaPdf(surf);
  result *= surf->evaluateSurfaceAreaPdf(dg);
  return result;
} // end SurfacePrimitiveList::evaluateSurfaceAreaPdf()

void SurfacePrimitiveList
   ::sampleSurfaceArea(const gpcpu::float4 *u,
                       const SurfacePrimitive **prims,
                       DifferentialGeometry *dg,
                       float *pdf,
                       const size_t n) const
{
  for(size_t i = 0; i != n; ++i)
  {
    sampleSurfaceArea(u[i][0], u[i][1],
                      u[i][2], u[i][3],
                      &prims[i], dg[i],
                      pdf[i]);
  } // end for i
} // end SurfacePrimitive::sampleSurfaceArea()

