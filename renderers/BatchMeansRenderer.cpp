/*! \file BatchMeansRenderer.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of BatchMeansRenderer class.
 */

#include "BatchMeansRenderer.h"
#include "../mutators/KelemenMutator.h"

BatchMeansRenderer
  ::BatchMeansRenderer(void)
    :Parent()
{
  ;
} // end BatchMeansRenderer::BatchMeansRenderer()

BatchMeansRenderer
  ::BatchMeansRenderer(const boost::shared_ptr<RandomSequence> &s,
                       const boost::shared_ptr<PathMutator> &mutator,
                       const boost::shared_ptr<ScalarImportance> &importance,
                       const size_t b)
    :Parent(s,mutator,importance),mTotalBatches(b)
{
  ;
} // end BatchMeansRenderer::BatchMeansRenderer()

void BatchMeansRenderer
  ::preprocess(void)
{
  Parent::preprocess();

  unsigned int w = dynamic_cast<RenderFilm*>(mRecord.get())->getWidth();
  unsigned int h = dynamic_cast<RenderFilm*>(mRecord.get())->getHeight();

  // zero the variance image
  mVarianceImage.resize(w,h);
  mVarianceImage.fill(Spectrum::black());
  mVarianceImage.setFilename("variance.exr");

  // zero the mean over batches image
  mMeanOverBatches.resize(w,h);
  mMeanOverBatches.fill(Spectrum::black());
  mMeanOverBatches.setFilename("meanoverbatches.exr");

  // zero the current batch
  mCurrentBatch.resize(w,h);
  mCurrentBatch.fill(Spectrum::black());

  mNumBatches = 0;
} // end BatchMeansRenderer::preprocess()

void BatchMeansRenderer
  ::postprocess(void)
{
  Parent::postprocess();

  // scale by 1 / (numBatches - 1)
  float s = 1.0f / (mNumBatches - 1);
  mVarianceImage.scale(Spectrum(s,s,s));
  mVarianceImage.postprocess();

  s = 1.0f / mNumBatches;
  mMeanOverBatches.scale(Spectrum(s,s,s));
  mMeanOverBatches.postprocess();
} // end BatchMeansRenderer::postprocess();

void BatchMeansRenderer
  ::kernel(ProgressCallback &progress)
{
  PathSampler::HyperPoint x, y;
  Path xPath, yPath;
  typedef std::vector<PathSampler::Result> ResultList;
  ResultList xResults, yResults;

  // estimate normalization constant and pick a seed
  float b = mImportance->estimateNormalizationConstant(mRandomSequence, mScene, mShadingContext, mMutator,
                                                       10000, mLocalPool, x, xPath);
  float invB = 1.0f / b;

  // initial seed
  mMutator->evaluate(xPath,xResults);
  float ix = (*mImportance)(x, xPath, xResults), iy = 0;
  float xPdf = ix * invB, yPdf = 0;

  // XXX this is still pretty gross
  float pLargeStep = 0;
  try
  {
    pLargeStep = dynamic_cast<KelemenMutator&>(*mMutator).getLargeStepProbability();
  } // end try
  catch(...)
  {
    ;
  } // end catch

  float a;
  int whichMutation;

  PathToImage mapToImage;

  // initialize the HaltCriterion
  // before we start rendering
  mHalt->init(this, &progress);

  // figure out how often to update the batch means variance
  size_t perBatchWork = progress.expected_count() / mTotalBatches;
  size_t timeToUpdateVariance = perBatchWork;

  // keep track of how many samples the current batch has received
  float batchSamples = 0;

  // main loop
  while(!(*mHalt)())
  {
    // mutate
    whichMutation = (*mMutator)(x,xPath,y,yPath);

    // evaluate
    if(whichMutation != -1)
    {
      yResults.clear();
      mMutator->evaluate(yPath, yResults);

      // compute importance
      iy = (*mImportance)(y, yPath, yResults);

      // compute pdf of y
      yPdf = iy * invB;
    } // end if
    else
    {
      iy = 0;
    } // end else

    // recompute x
    ix = (*mImportance)(x, xPath, xResults);

    // calculate accept probability
    a = 0;
    if(iy > 0)
    {
      a = mMutator->evaluateTransitionRatio(whichMutation, x, xPath, ix, y, yPath, iy);
      a = std::min<float>(1.0f, a * iy/ix);
    } // end if

    if(ix > 0)
    {
      // XXX TODO PERF: all this is redundant
      //          idea: accumulate x's weight and only deposit
      //                when a sample is finally rejected
      // record x
      float xWeight = (1.0f - a) / (xPdf+pLargeStep);
      mRecord->record(xWeight, x, xPath, xResults);

      // deposit to current batch as well
      mCurrentBatch.record(xWeight, x, xPath, xResults);

      // add to the acceptance image
      // XXX TODO: generalize this to all samplers somehow
      gpcpu::float2 pixel;
      float xu, xv;
      mapToImage(xResults[0], x, xPath, xu, xv);
      mAcceptanceImage.deposit(xu, xv, Spectrum(1.0f - a, 1.0f - a, 1.0f - a));
    } // end if

    if(iy > 0)
    {
      // record y
      float yWeight = (a + float(whichMutation))/(yPdf + pLargeStep);
      mRecord->record(yWeight, y, yPath, yResults);

      // deposit to current batch as well
      mCurrentBatch.record(yWeight, y, yPath, yResults);

      // add to the acceptance image
      // XXX TODO: generalize this to all samplers somehow
      float yu, yv;
      mapToImage(yResults[0], y, yPath, yu, yv);
      mAcceptanceImage.deposit(yu, yv, Spectrum(a, a, a));
    } // end if

    // accept?
    if((*mRandomSequence)() < a)
    {
      ++mNumAccepted;
      x = y;

      // safely copy the path
      copyPath(xPath, yPath);

      ix = iy;
      xResults = yResults;
      xPdf = yPdf;
    } // end if

    // purge all malloc'd memory for this sample
    mShadingContext->freeAll();

    ++mNumSamples;

    ++batchSamples;
    if(progress.count() >= timeToUpdateVariance)
    {
      ++mNumBatches;
      updateVarianceEstimate(batchSamples);

      // init a new batch
      mCurrentBatch.fill(Spectrum::black());

      // set a new time for an update
      timeToUpdateVariance += perBatchWork;

      // reset batchSamples
      batchSamples = 0;
    } // end if
  } // end for i

  // update the variance estimate one last time if we need to
  if(batchSamples != 0)
  {
    ++mNumBatches;
    updateVarianceEstimate(batchSamples);
  } // end if

  // purge the local store
  mLocalPool.freeAll();
} // end BatchMeansRenderer::kernel()

void BatchMeansRenderer
  ::updateVarianceEstimate(const float samplesPerBatch)
{
  std::cerr << "BatchMeansRenderer::updateVarianceEstimate(): " << std::endl;

  // divide by N to get a sample mean
  float spp = samplesPerBatch / (mCurrentBatch.getWidth() * mCurrentBatch.getHeight());
  mCurrentBatch.scale(Spectrum(1.0f / spp, 1.0f / spp, 1.0f / spp));

  char filename[33];
  sprintf(filename, "batch-%zd.exr", mNumBatches - 1);
  mCurrentBatch.writeEXR(filename);

  // for each pixel, update the current variance estimate
  for(size_t j = 0; j < mCurrentBatch.getHeight(); ++j)
  {
    for(size_t i = 0; i < mCurrentBatch.getWidth(); ++i)
    {
      // \from http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Algorithm_III
      
      Spectrum x = mCurrentBatch.raster(i,j);

      Spectrum delta;
      if(mNumBatches-1 > 0)
      {
        // get the old mean
        Spectrum oldMean = mMeanOverBatches.raster(i,j);
        delta = x - oldMean / (mNumBatches - 1);
      } // end if
      else
      {
        delta = x;
      } // end else
      
      // update the mean
      mMeanOverBatches.deposit(i,j,x);

      // update the variance over batches
      Spectrum newMean = mMeanOverBatches.raster(i,j) / mNumBatches;
      mVarianceImage.deposit(i,j, delta * (x - newMean));
    } // end for i
  } // end for j
} // end BatchMeansRenderer::updateVarianceEstimate()

