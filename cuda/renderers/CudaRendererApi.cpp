/*! \file CudaRendererApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of CudaRendererApi class.
 */

#include "CudaRendererApi.h"
#include "CudaDebugRenderer.h"
#include "CudaKajiyaPathTracer.h"
#include "../numeric/CudaRandomSequence.h"

using namespace boost;

void CudaRendererApi
  ::getDefaultAttributes(Gotham::AttributeMap &attr)
{
  attr["renderer:algorithm"] = "montecarlo";
  attr["renderer:threads"] = "1";
} // end CudaRendererApi::getDefaultAttributes()

CudaRenderer *CudaRendererApi
  ::renderer(Gotham::AttributeMap &attr,
             const Gotham::PhotonMaps &photonMaps)
{
  // create a RandomSequence
  shared_ptr<CudaRandomSequence> z(new CudaRandomSequence());

  // create a new CudaRenderer
  CudaRenderer *result = 0;

  // fish out the parameters
  std::string rendererName = attr["renderer:algorithm"];

  Gotham::AttributeMap::const_iterator a = attr.find("renderer:targetrays");
  if(a != attr.end())
  {
    std::cerr << "Warning: attribute \"renderer::targetrays\" is deprecated." << std::endl;
    std::cerr << "Please use \"renderer::target::function\" and \"renderer::target::count\" instead." << std::endl;

    attr["renderer:target:function"] = std::string("rays");
    attr["renderer:target:count"] = a->second;
  } // end if

  size_t numThreads = lexical_cast<size_t>(attr["renderer:threads"]);

  // create the renderer
  if(rendererName == "montecarlo")
  {
    CudaKajiyaPathTracer *r = new CudaKajiyaPathTracer();
    r->setWorkBatchSize(numThreads);
    r->setRandomSequence(shared_ptr<CudaRandomSequence>(new CudaRandomSequence()));
    result = r;
  } // end if
  else if(rendererName == "debug")
  {
    CudaDebugRenderer *r = new CudaDebugRenderer();
    r->setWorkBatchSize(numThreads);
    result = r;
  } // end else if
  else
  {
    std::cerr << "Warning: unknown rendering algorithm \"" << rendererName << "\"." << std::endl;

    // just do kajiya
    CudaKajiyaPathTracer *r = new CudaKajiyaPathTracer();
    r->setWorkBatchSize(numThreads);
    r->setRandomSequence(shared_ptr<CudaRandomSequence>(new CudaRandomSequence()));
    result = r;
  } // end else

  // XXX Remove this when we have successfully generalized target counts
  // get spp
  size_t spp = 4;
  a = attr.find("renderer:spp");
  if(a != attr.end())
  {
    spp = lexical_cast<size_t>(a->second);
  } // end if

  result->setSamplesPerPixel(spp);

  return result;
} // end RendererApi::renderer()

