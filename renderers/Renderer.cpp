/*! \file Renderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of Renderer class.
 */

#include "Renderer.h"
#include <boost/date_time/posix_time/posix_time.hpp>
#include "../primitives/Scene.h"
#include <iostream>
#include "../records/RenderFilm.h"

using namespace boost;

// XXX this is so friggin ugly
std::ostream &silence(std::ostream &os)
{
  os.setstate(std::ios_base::badbit);
  return os;
} // end silence()

Renderer::ProgressCallback
  ::ProgressCallback(void)
    :Parent(ULONG_MAX, silence(std::cout))
{
  // XXX this hacks around boost::progress_display's
  //     insistence on outputting as soon as it's
  //     constructed

  // this line turns the os back on
  std::cout.clear(std::ios_base::goodbit);
} // end ProgressCallback::ProgressCallback()

void Renderer::ProgressCallback
  ::restart(unsigned long expected_count)
{
  if(expected_count != ULONG_MAX)
  {
    Parent::restart(expected_count);
  } // end 
} // end ProgressCallback::restart()

void Renderer
  ::render(ProgressCallback &progress)
{
  preprocess();
  kernel(progress);
  postprocess();
} // end Renderer::render()

void Renderer
  ::preprocess(void)
{
  // preprocess the record
  mRecord->preprocess();

  // start the timer
  mTimer.restart();
} // end Renderer::preprocess()

void Renderer
  ::postprocess(void)
{
  // stop the timer
  double elapsed = mTimer.elapsed();

  postRenderReport(elapsed);

  mRecord->postprocess();
} // end Renderer::postprocess()

void Renderer
  ::postRenderReport(const double elapsed) const
{
  std::cout << "Finished." << std::endl;

  unsigned long minutes = static_cast<unsigned long>(elapsed) % 60;
  unsigned long hours = static_cast<unsigned long>(minutes) % (60*60);
  unsigned long seconds = static_cast<unsigned long>(elapsed) - hours*60*60 - minutes*60;

  boost::posix_time::time_duration td(hours,minutes,seconds);
  std::cout << "Seconds elapsed: " << elapsed << std::endl;
  //std::cout << "Time elapsed: " << td << std::endl;
  std::cout << "Rays cast: " << mScene->getRaysCast() << std::endl;
  std::cout << "Shadow rays cast: " << mScene->getShadowRaysCast() << std::endl;
  std::cout << "Blocked shadow rays: " << mScene->getBlockedShadowRays() << std::endl;
  std::cout << "Shadow rate: " << static_cast<float>(mScene->getBlockedShadowRays()) / static_cast<float>(mScene->getShadowRaysCast()) << std::endl;
  std::cout << "Rays per second: " << static_cast<double>(mScene->getRaysCast()) / elapsed << std::endl;
} // end Renderer::postRenderReport()

std::string Renderer
  ::getRenderParameters(void) const
{
  return std::string("");
} // end Renderer::getRenderParameters()

