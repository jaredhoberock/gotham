/*! \file RecursiveMetropolisRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RecursiveMetropolisRenderer class.
 */

#include "RecursiveMetropolisRenderer.h"
#include "../importance/EstimateImportance.h"
#include "../importance/LuminanceImportance.h"

RecursiveMetropolisRenderer
  ::RecursiveMetropolisRenderer(void)
    :Parent()
{
  ;
} // end RecursiveMetropolisRenderer::RecursiveMetropolisRenderer()

RecursiveMetropolisRenderer
  ::RecursiveMetropolisRenderer(const boost::shared_ptr<RandomSequence> &s,
                                const boost::shared_ptr<PathMutator> &mutator,
                                const unsigned int target)
    :Parent(s,mutator,boost::shared_ptr<ScalarImportance>((ScalarImportance*)0),target),mRecursionScale(0.5f)
{
  ;
} // end RecursiveMetropolisRenderer::RecursiveMetropolisRenderer()

RecursiveMetropolisRenderer
  ::RecursiveMetropolisRenderer(boost::shared_ptr<const Scene> &s,
                                boost::shared_ptr<Record> &r,
                                const boost::shared_ptr<RandomSequence> &sequence,
                                const boost::shared_ptr<PathMutator> &m,
                                const unsigned int target)
    :Parent(s,r,sequence,m,boost::shared_ptr<ScalarImportance>((ScalarImportance*)0),target),mRecursionScale(0.5f)
{
  ;
} // end RecursiveMetropolisRenderer::RecursiveMetropolisRenderer()

void RecursiveMetropolisRenderer
  ::kernel(ProgressCallback &progress)
{
  std::cerr << "RecursiveMetropolisRenderer::kernel(): entered." << std::endl;

  // render a half-res version of the Image first
  // XXX is there a way to do this with Records in general?
  size_t w = static_cast<size_t>(mRecursionScale * dynamic_cast<RenderFilm*>(mRecord.get())->getWidth());
  size_t h = static_cast<size_t>(mRecursionScale * dynamic_cast<RenderFilm*>(mRecord.get())->getHeight());
  Target target = static_cast<Target>(mRecursionScale * mRecursionScale * Parent::mRayTarget);

  ScalarImportance *importance = 0;
  if(w > 0 && h > 0 && target > 1000000)
  {
    // recurse
    std::cerr << "recursing with " << target << " target rays." << std::endl;
    char buf[33];
    sprintf(buf, "estimate-%dx%d.exr", (int)w, (int)h);
    boost::shared_ptr<Record> lowResImage(new RenderFilm(w,h,buf));
    RecursiveMetropolisRenderer recurse(mScene,
                                        lowResImage,
                                        mRandomSequence,
                                        mMutator,
                                        target);

    // render
    ProgressCallback p;
    recurse.render(p);

    // use an estimate for importance
    importance = new EstimateImportance(*boost::dynamic_pointer_cast<RandomAccessFilm,Record>(lowResImage));
  } // end if
  else
  {
    std::cerr << "terminating." << std::endl;
    std::cerr << "w: " << w << std::endl;
    std::cerr << "h: " << h << std::endl;
    std::cerr << "target: " << target << std::endl;

    // the importance function is just luminance
    importance = new LuminanceImportance();
  } // end else

  mImportance.reset(importance);

  // preprocess again
  // XXX this is kind of nasty
  preprocess();

  // defer to the Parent
  Parent::kernel(progress);
} // end RecursiveMetropolisRenderer::kernel()

void RecursiveMetropolisRenderer
  ::preprocess(void)
{
  // bypass MetropolisRenderer
  MonteCarloRenderer::preprocess();

  // zero the accepted count
  mNumAccepted = 0;

  // XXX kill this nastiness somehow
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());
  if(film != 0)
  {
    // zero the acceptance image
    mAcceptanceImage.resize(film->getWidth(), film->getHeight());
    char buf[33];
    sprintf(buf, "acceptance-%dx%d.exr", film->getWidth(), film->getHeight());
    //mAcceptanceImage.setFilename("acceptance.exr");
    mAcceptanceImage.setFilename(std::string(buf));
    mAcceptanceImage.preprocess();
  } // end if

  // preprocess the mutator
  mMutator->preprocess();

  // preprocess the scalar importance if it isn't null
  if(mImportance.get() != 0)
    mImportance->preprocess(mRandomSequence, mScene, mMutator, *this);
} // end RecursiveMetropolisRenderer::preprocess()

