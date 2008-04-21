/*! \file CudaDebugRenderer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for CudaDebugRenderer.h.
 */

#include "CudaDebugRenderer.h"

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

