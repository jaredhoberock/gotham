/*! \file SIMDRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of SIMDRenderer class.
 */

#include "SIMDRenderer.h"
#include "../records/RenderFilm.h"

void SIMDRenderer
  ::kernel(ProgressCallback &progress)
{
  // XXX TODO: kill this
  RenderFilm *film = dynamic_cast<RenderFilm*>(mRecord.get());

  // compute the total work
  size_t totalWork = film->getWidth() * film->getHeight();

  // compute the total number of batches we need to complete the work
  size_t numBatches = totalWork / mWorkBatchSize;

  // work on full batches
  for(size_t i = 0;
      i != numBatches;
      ++i)
  {
    for(size_t thread = 0;
        thread != mWorkBatchSize;
        ++thread)
    {
      // work on thread i
      kernel(i * mWorkBatchSize + thread);
    } // end for thread

    // purge all malloc'd memory for this batch
    ScatteringDistributionFunction::mPool.freeAll();

    // update progress en masse
    progress += mWorkBatchSize;
  } // end for i

  // finish the last partial batch
  for(size_t i = 0;
      i != totalWork % mWorkBatchSize;
      ++i)
  {
    // work on thread
    kernel(numBatches * mWorkBatchSize + i);

    ++progress;
  } // end if

  // purge all malloc'd memory for the partial batch
  ScatteringDistributionFunction::mPool.freeAll();
} // end SIMDRenderer::kernel()

