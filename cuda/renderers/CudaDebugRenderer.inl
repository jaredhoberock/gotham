/*! \file CudaDebugRenderer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CudaDebugRenderer.h.
 */

#include "CudaDebugRenderer.h"
#include "../primitives/CudaScene.h"
#include "../shading/CudaShadingContext.h"

CudaDebugRenderer
  ::CudaDebugRenderer(void)
    :Parent()
{
  ;
} // end CudaDebugRenderer::CudaDebugRenderer()

CudaDebugRenderer
  ::CudaDebugRenderer(boost::shared_ptr<const Scene> s,
                      boost::shared_ptr<Record> r)
    :Parent(s,r)
{
  ;
} // end CudaDebugRenderer::CudaDebugRenderer()

void CudaDebugRenderer
  ::setScene(const boost::shared_ptr<const Scene> &s)
{
  if(dynamic_cast<const CudaScene*>(s.get()))
  {
    Parent::setScene(s);
  } // end if
  else
  {
    std::cerr << "CudaDebugRenderer::setScene(): scene must be a CudaScene!" << std::endl;
    exit(-1);
  } // end else
} // end CudaDebugRenderer::setScene()

void CudaDebugRenderer
  ::setShadingContext(const boost::shared_ptr<ShadingContext> &s)
{
  if(dynamic_cast<CudaShadingContext*>(s.get()))
  {
    Parent::setShadingContext(s);
  } // end if
  else
  {
    std::cerr << "CudaDebugRenderer::setShadingContext(): scene must be a CudaShadingContext!" << std::endl;
    exit(-1);
  } // end else
} // end CudaDebugRenderer::setShadingContext()

