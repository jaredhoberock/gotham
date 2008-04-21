/*! \file CudaRenderer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CudaRenderer.h.
 */

#include "CudaRenderer.h"
#include "../primitives/CudaScene.h"
#include "../shading/CudaShadingContext.h"

CudaRenderer
  ::CudaRenderer(void)
    :Parent()
{
  ;
} // end CudaRenderer::CudaRenderer()

CudaRenderer
  ::CudaRenderer(boost::shared_ptr<const Scene> s,
                 boost::shared_ptr<Record> r)
  :Parent(s,r)
{
  ;
} // end CudaRenderer::CudaRenderer()

void CudaRenderer
  ::setScene(const boost::shared_ptr<const Scene> &s)
{
  if(dynamic_cast<const CudaScene*>(s.get()))
  {
    Parent::setScene(s);
  } // end if
  else
  {
    std::cerr << "CudaRenderer::setScene(): scene must be a CudaScene!" << std::endl;
    exit(-1);
  } // end else
} // end CudaRenderer::setScene()

void CudaRenderer
  ::setShadingContext(const boost::shared_ptr<ShadingContext> &s)
{
  if(dynamic_cast<CudaShadingContext*>(s.get()))
  {
    Parent::setShadingContext(s);
  } // end if
  else
  {
    std::cerr << "CudaRenderer::setShadingContext(): scene must be a CudaShadingContext!" << std::endl;
    exit(-1);
  } // end else
} // end CudaRenderer::setShadingContext()

