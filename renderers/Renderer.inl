/*! \file Renderer.inl
 *  \author Jared Hoberock
 *  \brief Inline file for Renderer.h.
 */

#include "Renderer.h"

Renderer
  ::Renderer(void)
    :mScene(),mRecord(),mSamplesPerPixel(1)
{
  ;
} // end Renderer::Renderer()

Renderer
  ::Renderer(boost::shared_ptr<const Scene> s,
             boost::shared_ptr<Record> r)
    :mScene(s),mRecord(r)
{
  ;
} // end Renderer::Renderer()

Renderer
  ::~Renderer(void)
{
  ;
} // end Renderer::~Renderer()

void Renderer
  ::setScene(const boost::shared_ptr<const Scene> &s)
{
  mScene = s;
} // end Renderer::setScene()

boost::shared_ptr<const Scene> Renderer
  ::getScene(void) const
{
  return mScene;
} // end Renderer::getScene()

void Renderer
  ::setRecord(boost::shared_ptr<Record> r)
{
  mRecord = r;
} // end Renderer::setRecord()

boost::shared_ptr<const Record> Renderer
  ::getRecord(void) const
{
  return mRecord;
} // end Renderer::getRecord()

void Renderer
  ::setShadingContext(const boost::shared_ptr<ShadingContext> &s)
{
  mShadingContext = s;
} // end Renderer::setShadingContext()

const ShadingContext &Renderer
  ::getShadingContext(void) const
{
  return *mShadingContext;
} // end Renderer::setShadingContext()

ShadingContext &Renderer
  ::getShadingContext(void)
{
  return *mShadingContext;
} // end Renderer::setShadingContext()

void Renderer
  ::setSamplesPerPixel(const unsigned int spp)
{
  mSamplesPerPixel = spp;
} // end Renderer::setSamplesPerPixel()

