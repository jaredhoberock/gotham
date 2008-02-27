/*! \file SIMDRenderer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for SIMDRenderer.h.
 */

#include "SIMDRenderer.h"

SIMDRenderer
  ::SIMDRenderer(void)
    :Parent(),mWorkBatchSize(2)
{
  ;
} // end SIMDRenderer::SIMDRenderer()

SIMDRenderer
  ::SIMDRenderer(boost::shared_ptr<const Scene> s,
                 boost::shared_ptr<Record> r)
    :Parent(s,r),mWorkBatchSize(2)
{
  ;
} // end SIMDRenderer::SIMDRenderer()

void SIMDRenderer
  ::setWorkBatchSize(const size_t n)
{
  mWorkBatchSize = n;
} // end SIMDRenderer::setWorkBatchSize()

