#include "../records/RandomAccessFilm.h"

int main(int argc, char **argv)
{
  if(argc < 3)
  {
    std::cerr << "usage: imgclean in.exr out.exr" << std::endl;
    exit(-1);
  } // end if

  RandomAccessFilm in;
  in.readEXR(argv[1]);

  RandomAccessFilm out;
  out = in;

  // first set any nans to inf
  for(size_t y = 0; y != out.getHeight(); ++y)
  {
    for(size_t x = 0; x != out.getWidth(); ++x)
    {
      for(size_t c = 0; c != Spectrum::numElements(); ++c)
      {
        if(out.raster(x,y)[c] != out.raster(x,y)[c])
        {
          out.raster(x,y)[c] = std::numeric_limits<float>::infinity();
        } // end if
      } // end for c
    } // end for x
  } // end for y

  // now erode infs
  while(out.erode(std::numeric_limits<float>::infinity()));

  // write output
  out.writeEXR(argv[2]);

  return 0;
} // end main()

