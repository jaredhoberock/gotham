/*! \file RendererApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RendererApi class.
 */

#include "RendererApi.h"
#include "PathDebugRenderer.h"
#include "EnergyRedistributionRenderer.h"
#include "MetropolisRenderer.h"
#include "DebugRenderer.h"
#include "TargetRaysRenderer.h"
#include "RecursiveMetropolisRenderer.h"
#include "VarianceRenderer.h"
#include "BatchMeansRenderer.h"
#include "../path/PathApi.h"
#include "../mutators/MutatorApi.h"
#include "../importance/ImportanceApi.h"
#include "../importance/LuminanceImportance.h"

using namespace boost;

Renderer *RendererApi
  ::renderer(const Gotham::AttributeMap &attr)
{
  // create a RandomSequence
  shared_ptr<RandomSequence> z(new RandomSequence());

  // create a new Renderer
  Renderer *result = 0;
  std::string rendererName = "montecarlo";

  // fish out the parameters
  Gotham::AttributeMap::const_iterator a = attr.find("renderer::algorithm");
  if(a != attr.end())
  {
    any val = a->second;
    rendererName = boost::any_cast<std::string>(val);
  } // end if

  float k = 1;
  a = attr.find("renderer::energyredistribution::mutationspersample");
  if(a != attr.end())
  {
    any val = a->second;
    k = static_cast<float>(atof(boost::any_cast<std::string>(val).c_str()));
  } // end if

  unsigned int m = 100;
  a = attr.find("renderer::energyredistribution::chainlength");
  if(a != attr.end())
  {
    any val = a->second;
    m = atoi(boost::any_cast<std::string>(val).c_str());
  } // end if

  unsigned int targetRays = 0;
  a = attr.find("renderer::targetrays");
  if(a != attr.end())
  {
    any val = a->second;
    targetRays = atoi(boost::any_cast<std::string>(val).c_str());
  } // end if

  // create the renderer
  if(rendererName == "montecarlo")
  {
    // create a PathSampler
    shared_ptr<PathSampler> sampler(PathApi::sampler(attr));

    result = new PathDebugRenderer(z, sampler);
  } // end if
  else if(rendererName == "energyredistribution")
  {
    // create a PathMutator
    shared_ptr<PathMutator> mutator(MutatorApi::mutator(attr));

    // create a ScalarImportance
    shared_ptr<ScalarImportance> importance(ImportanceApi::importance(attr));
    result = new EnergyRedistributionRenderer(k, m, z, mutator, importance);
  } // end else if
  else if(rendererName == "metropolis")
  {
    // create a PathMutator
    shared_ptr<PathMutator> mutator(MutatorApi::mutator(attr));

    // create a ScalarImportance
    shared_ptr<ScalarImportance> importance(ImportanceApi::importance(attr));

    if(targetRays != 0)
    {
      result = new TargetRaysRenderer(z, mutator, importance, targetRays);
    } // end if
    else
    {
      result = new MetropolisRenderer(z, mutator, importance);
    } // end else
  } // end else if
  else if(rendererName == "recursivemetropolis")
  {
    // create a PathMutator
    shared_ptr<PathMutator> mutator(MutatorApi::mutator(attr));

    result = new RecursiveMetropolisRenderer(z, mutator, targetRays);
  } // end else if
  else if(rendererName == "variance")
  {
    // create a PathSampler
    shared_ptr<PathSampler> sampler(PathApi::sampler(attr));

    VarianceRenderer *vr = new VarianceRenderer(z, sampler);
    result = vr;
    boost::shared_ptr<Record> v(new RenderFilm(512, 512, "variance.exr"));
    vr->setVarianceRecord(v);
  } // end else if
  else if(rendererName == "batchmeans")
  {
    // create a PathMutator
    shared_ptr<PathMutator> mutator(MutatorApi::mutator(attr));

    // create a ScalarImportance
    shared_ptr<ScalarImportance> importance(ImportanceApi::importance(attr));

    result = new BatchMeansRenderer(z, mutator, importance);
  } // end else if
  else if(rendererName == "debug")
  {
    result = new DebugRenderer();
  } // end else if
  else
  {
    std::cerr << "Warning: unknown rendering algorithm \"" << rendererName << "\"." << std::endl;

    // create a PathSampler
    shared_ptr<PathSampler> sampler(PathApi::sampler(attr));
    result = new PathDebugRenderer(z, sampler);
  } // end else

  // get spp
  unsigned int spp = 4;
  a = attr.find("renderer::spp");
  if(a != attr.end())
  {
    any val = a->second;
    spp = atoi(any_cast<std::string>(val).c_str());
  } // end if

  result->setSamplesPerPixel(spp);

  return result;
} // end RendererApi::renderer()

