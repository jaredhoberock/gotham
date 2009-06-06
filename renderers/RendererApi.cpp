/*! \file RendererApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RendererApi class.
 */

#include "RendererApi.h"
#include "PathDebugRenderer.h"
#include "EnergyRedistributionRenderer.h"
#include "MetropolisRenderer.h"
#include "DebugRenderer.h"
#include "SIMDDebugRenderer.h"
#include "MultiStageMetropolisRenderer.h"
#include "VarianceRenderer.h"
#include "BatchMeansRenderer.h"
#include "NoiseAwareMetropolisRenderer.h"
#include "../path/PathApi.h"
#include "../mutators/MutatorApi.h"
#include "../importance/ImportanceApi.h"
#include "../importance/LuminanceImportance.h"
#include "HaltCriterion.h"

#include <boost/tuple/tuple.hpp>
#include <boost/python/exec.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/tuple.hpp>

using namespace boost;
using namespace boost::python;

void RendererApi
  ::getDefaultAttributes(Gotham::AttributeMap &attr)
{
  // call HaltCriterion first
  HaltCriterion::getDefaultAttributes(attr);

  attr["renderer:algorithm"] = "montecarlo";
  attr["renderer:energyredistribution:mutationspersample"] = "1";
  attr["renderer:energyredistribution:chainlength"] = "100";
  attr["renderer:batchmeans:batches"] = "2";
  attr["renderer:noiseawaremetropolis:varianceexponent"] = "0.5";
  attr["renderer:seed"] = "13";
} // end RendererApi::getDefaultAttributes()

Renderer *RendererApi
  ::renderer(Gotham::AttributeMap &attr,
             const Gotham::PhotonMaps &photonMaps)
{
  // create a RandomSequence
  unsigned int seed = lexical_cast<unsigned int>(attr["renderer:seed"]);
  shared_ptr<RandomSequence> z(new RandomSequence(seed));

  // create a new Renderer
  Renderer *result = 0;

  // fish out the parameters
  std::string rendererName = attr["renderer:algorithm"];

  float k = lexical_cast<float>(attr["renderer:energyredistribution:mutationspersample"]);

  size_t m = lexical_cast<size_t>(attr["renderer:energyredistribution:chainlength"]);

  size_t numBatches = lexical_cast<size_t>(attr["renderer:batchmeans:batches"]);

  Gotham::AttributeMap::const_iterator a = attr.find("renderer:targetrays");
  if(a != attr.end())
  {
    std::cerr << "Warning: attribute \"renderer::targetrays\" is deprecated." << std::endl;
    std::cerr << "Please use \"renderer::target::function\" and \"renderer::target::count\" instead." << std::endl;

    attr["renderer:target:function"] = std::string("rays");
    attr["renderer:target:count"] = a->second;
  } // end if

  float varianceExponent = lexical_cast<float>(attr["renderer:noiseawaremetropolis:varianceexponent"]);

  std::string acceptanceFilename = attr["record:acceptance:outfile"];
  std::string proposalFilename   = attr["record:proposals:outfile"];
  std::string targetFilename     = attr["record:target:outfile"];

  // create the renderer
  if(rendererName == "montecarlo")
  {
    // create a PathSampler
    shared_ptr<PathSampler> sampler(PathApi::sampler(attr, photonMaps));

    result = new PathDebugRenderer(z, sampler);
  } // end if
  else if(rendererName == "energyredistribution")
  {
    // create a PathMutator
    shared_ptr<PathMutator> mutator(MutatorApi::mutator(attr, photonMaps));

    // create a ScalarImportance
    shared_ptr<ScalarImportance> importance(ImportanceApi::importance(attr));
    result = new EnergyRedistributionRenderer(k, m, z, mutator, importance);
  } // end else if
  else if(rendererName == "metropolis")
  {
    // create a PathMutator
    shared_ptr<PathMutator> mutator(MutatorApi::mutator(attr, photonMaps));

    // create a ScalarImportance
    shared_ptr<ScalarImportance> importance(ImportanceApi::importance(attr));

    result = new MetropolisRenderer(z, mutator, importance);
  } // end else if
  else if(rendererName == "multistagemetropolis")
  {
    // create a PathMutator
    shared_ptr<PathMutator> mutator(MutatorApi::mutator(attr, photonMaps));

    // create a ScalarImportance
    shared_ptr<ScalarImportance> importance(ImportanceApi::importance(attr));

    result = new MultiStageMetropolisRenderer(z, mutator, importance);
  } // end else if
  else if(rendererName == "noiseawaremetropolis")
  {
    // create a PathMutator
    shared_ptr<PathMutator> mutator(MutatorApi::mutator(attr, photonMaps));

    // create a ScalarImportance
    shared_ptr<ScalarImportance> importance(ImportanceApi::importance(attr));

    result = new NoiseAwareMetropolisRenderer(z, mutator, importance, varianceExponent);
    static_cast<NoiseAwareMetropolisRenderer*>(result)->setTargetFilename(targetFilename);
  } // end else if
  else if(rendererName == "altnoiseawaremetropolis")
  {
    std::cerr << "Warning: rendering algorithm \"altnoiseawaremetropolis\" is deprecated. Please use \"noiseawaremetropolis\" instead." << std::endl;

    // create a PathMutator
    shared_ptr<PathMutator> mutator(MutatorApi::mutator(attr, photonMaps));

    // create a ScalarImportance
    shared_ptr<ScalarImportance> importance(ImportanceApi::importance(attr));

    result = new NoiseAwareMetropolisRenderer(z, mutator, importance, varianceExponent);
    static_cast<NoiseAwareMetropolisRenderer*>(result)->setTargetFilename(targetFilename);
  } // end else if
  else if(rendererName == "variance")
  {
    // create a PathSampler
    shared_ptr<PathSampler> sampler(PathApi::sampler(attr, photonMaps));

    VarianceRenderer *vr = new VarianceRenderer(z, sampler);
    result = vr;
    boost::shared_ptr<Record> v(new RenderFilm(512, 512, "variance.exr"));
    vr->setVarianceRecord(v);
  } // end else if
  else if(rendererName == "batchmeans")
  {
    // create a PathMutator
    shared_ptr<PathMutator> mutator(MutatorApi::mutator(attr, photonMaps));

    // create a ScalarImportance
    shared_ptr<ScalarImportance> importance(ImportanceApi::importance(attr));

    result = new BatchMeansRenderer(z, mutator, importance, numBatches);
  } // end else if
  else if(rendererName == "debug")
  {
    boost::tuple<size_t, size_t> spp(1,1);

    Gotham::AttributeMap::const_iterator a = attr.find("renderer:spp");
    if(a != attr.end())
    {
      try
      {
        // try to convert a tuple by evaluating the python expression
        python::tuple temp = extract<python::tuple>(eval(a->second.c_str()));
        size_t sx = extract<size_t>(temp[0]);
        size_t sy = extract<size_t>(temp[1]);
        spp = boost::make_tuple(sx,sy);
      } // end try
      catch(...)
      {
        try
        {
          size_t xStrata = lexical_cast<size_t>(a->second);
          spp = boost::tuple<size_t,size_t>(xStrata,xStrata);
        } // end try
        catch(bad_lexical_cast &e)
        {
          std::cerr << "RendererApi::renderer(): Warning: Couldn't interpret " << a->second << " as samples per pixel (xStrata,yStrata)." << std::endl;
        } // end catch
      } // end catch
    } // end if

    result = new DebugRenderer(spp.get<0>(),spp.get<1>());
  } // end else if
  else
  {
    std::cerr << "Warning: unknown rendering algorithm \"" << rendererName << "\"." << std::endl;

    // create a PathSampler
    shared_ptr<PathSampler> sampler(PathApi::sampler(attr, photonMaps));
    result = new PathDebugRenderer(z, sampler);
  } // end else

  // create a HaltCriterion for MonteCarloRenderers
  if(dynamic_cast<MonteCarloRenderer*>(result))
  {
    shared_ptr<HaltCriterion> halt(HaltCriterion::createCriterion(attr));
    dynamic_cast<MonteCarloRenderer*>(result)->setHaltCriterion(halt);
  } // end if

  // set the acceptance filename for MetropolisRenderers
  // XXX this is all so shitty
  //     i guess we need a MetropolisRecord which can also integrate acceptance and proposals?
  if(dynamic_cast<MetropolisRenderer*>(result))
  {
    MetropolisRenderer *r = dynamic_cast<MetropolisRenderer*>(result);
    r->setAcceptanceFilename(acceptanceFilename);
    r->setProposalFilename(proposalFilename);
  } // end try

  return result;
} // end RendererApi::renderer()

