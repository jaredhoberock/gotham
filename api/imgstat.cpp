#include "../records/RandomAccessFilm.h"

void reportVisualError(const RandomAccessFilm &in,
                       const RandomAccessFilm &ref)
{
  float displayMax = 100.0f;

  // scale to display max
  RandomAccessFilm displayIn = in;
  displayIn.scale(Spectrum(displayMax, displayMax, displayMax));

  RandomAccessFilm displayRef = ref;
  displayRef.scale(Spectrum(displayMax, displayMax, displayMax));

  // compute target image
  RandomAccessFilm target(ref.getWidth(), ref.getHeight());
  for(size_t y = 0; y != target.getHeight(); ++y)
  {
    for(size_t x = 0; x != target.getWidth(); ++x)
    {
      float e = fabs(displayRef.raster(x,y).luminance() - displayIn.raster(x,y).luminance());
      target.raster(x,y) = Spectrum::white();
      target.raster(x,y).setLuminance(e);
    } // end for x
  } // end for y

  RandomAccessFilm tvi = ref;
  tvi.applyThresholdVersusIntensityFunction();
  
  // divide by tvi
  target.divideLuminance(tvi, 0);

  // now add everything up
  float sum = 0;
  for(size_t y = 0; y != target.getHeight(); ++y)
  {
    for(size_t x = 0; x != target.getWidth(); ++x)
    {
      sum += target.raster(x,y).luminance();
    } // end for x
  } // end for y
  sum /= (target.getWidth() * target.getHeight());

  std::cerr << "Visual error: " << sum << std::endl;
} // end reportVisualError()

int main(int argc, char **argv)
{
  if(argc < 2)
  {
    std::cerr << "usage: imgstat in.exr [ref.exr]" << std::endl;
    exit(-1);
  } // end if

  RandomAccessFilm in;
  in.readEXR(argv[1]);

  // first compute mean
  Spectrum mean(Spectrum::black());
  for(size_t y = 0; y != in.getHeight(); ++y)
  {
    for(size_t x = 0; x != in.getWidth(); ++x)
    {
      mean += in.raster(x,y);
    } // end for x
  } // end for y

  mean /= (in.getWidth() * in.getHeight());

  // now compute variance
  float variance = 0;
  for(size_t y = 0; y != in.getHeight(); ++y)
  {
    for(size_t x = 0; x != in.getWidth(); ++x)
    {
      float diff2 = in.raster(x,y).luminance() - mean.luminance();
      diff2 *= diff2;
      variance += diff2;
    } // end for x
  } // end for y

  variance /= (in.getWidth() * in.getHeight());

  // now compute skew & kurtosis
  float kurtosisNumerator = 0;
  float skewNumerator = 0;
  float denom = 0;
  for(size_t y = 0; y != in.getHeight(); ++y)
  {
    for(size_t x = 0; x != in.getWidth(); ++x)
    {
      float diff = in.raster(x,y).luminance() - mean.luminance();
      float diff2 = diff * diff;
      float diff3 = diff * diff2;
      float diff4 = diff2 * diff2;

      skewNumerator += diff3;
      kurtosisNumerator += diff4;
      denom += diff2;
    } // end for x
  } // end for y

  skewNumerator *= sqrtf(static_cast<float>(in.getWidth() * in.getHeight()));
  kurtosisNumerator *= (in.getWidth() * in.getHeight());

  float skew = skewNumerator / powf(denom,1.5f);
  float kurtosis = kurtosisNumerator / powf(denom,2.0f) - 3.0f;

  std::cout << "Filename: " << argv[0] << std::endl;
  std::cout << "mean: " << mean << std::endl;
  std::cout << "variance: " << variance << std::endl;
  std::cout << "skew: " << skew << std::endl;
  std::cout << "kurtosis: " << kurtosis << std::endl;

  if(argc > 2)
  {
    RandomAccessFilm ref;
    ref.readEXR(argv[2]);
    if(ref.getWidth() != in.getWidth() ||
       ref.getHeight() != in.getHeight())
    {
      RandomAccessFilm resampled(in.getWidth(), in.getHeight());
      ref.resample(resampled);
      ref = resampled;
    } // end if

    // downsample then upsample the reference
    RandomAccessFilm down(ref.getWidth() / 4, ref.getHeight() / 4);
    ref.resample(down);
    down.resample(ref);

    // first normalize the input's mean luminance to match the reference
    float referenceLuminance = ref.computeMeanLuminance();
    float s = referenceLuminance / in.computeMeanLuminance();
    in.scale(Spectrum(s,s,s));

    // measure squared error
    float squaredError = 0;
    for(size_t y = 0; y != in.getHeight(); ++y)
    {
      for(size_t x = 0; x != in.getWidth(); ++x)
      {
        float r = ref.raster(x,y).luminance();

        if(r == r && r != std::numeric_limits<float>::infinity())
        {
          float i = in.raster(x,y).luminance();
          //i = std::min(powf(i,1.0f / 2.1f), 1.0f);
          //r = std::min(powf(i,1.0f / 2.1f), 1.0f);

          float diff = i - r;
          squaredError += diff * diff;
        } // end if
        else
        {
          std::cerr << "Warning: skipped inf or nan in reference at " << x << "," << y << std::endl;
        } // end else
      } // end for x
    } // end for y
    squaredError /= (in.getWidth() * in.getHeight());

    // measure relative error
    float relError = 0;
    for(size_t y = 0; y != in.getHeight(); ++y)
    {
      for(size_t x = 0; x != in.getWidth(); ++x)
      {
        float r = ref.raster(x,y).luminance();

        if(r == r && r != std::numeric_limits<float>::infinity())
        {

          float i = in.raster(x,y).luminance();

          float diff = fabs(i - r);

          diff /= std::max(0.0005f, r);

          relError += diff;
        } // end if
        else
        {
          std::cerr << "Warning: skipped inf or nan in reference at " << x << "," << y << std::endl;
        } // end else
      } // end for x
    } // end for y
    relError /= (in.getWidth() * in.getHeight());

    std::cout << "squared error: " << squaredError << std::endl;
    std::cout << "relative error: " << relError << std::endl;


    reportVisualError(in, ref);
  } // end if

  return 0;
} // end main()

