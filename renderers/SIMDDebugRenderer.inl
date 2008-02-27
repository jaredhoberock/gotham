/*! \file SIMDDebugRenderer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for SIMDDebugRenderer.h.
 */

#include "SIMDDebugRenderer.h"

SIMDDebugRenderer
  ::SIMDDebugRenderer(void)
    :Parent()
{
  ;
} // end SIMDDebugRenderer::SIMDDebugRenderer()

SIMDDebugRenderer
  ::SIMDDebugRenderer(boost::shared_ptr<const Scene> s,
                      boost::shared_ptr<Record> r)
    :Parent(s,r)
{
  ;
} // end SIMDDebugRenderer::SIMDDebugRenderer()

