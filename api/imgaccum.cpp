#include <string>
#include "../records/RandomAccessFilm.h"

int main(int argc, char **argv)
{
  if(argc < 2)
  {
    std::cerr << "usage: imgaccum in.exr [in2.exr in3.exr ...]" << std::endl;
    exit(-1);
  } // end if

  RandomAccessFilm in;

  // read the first image
  in.readEXR(argv[1]);

  RandomAccessFilm out;
  out = in;

  for(int i = 1; i < argc; ++i)
  {
    // read each successive image
    in.readEXR(argv[i]);

    assert(in.getWidth() == out.getWidth());
    assert(in.getHeight() == out.getHeight());

    // accumulate each pixel to out
    for(size_t y = 0; y != out.getHeight(); ++y)
    {
      for(size_t x = 0; x != out.getWidth(); ++x)
      {
        out.raster(x,y) += in.raster(x,y);
      } // end for x
    } // end for y
  } // end for i

  // scale out by 1/N
  float w = 1.0f / (argc - 1);
  Spectrum s(w,w,w);
  out.scale(s);

  // write output
  out.writeEXR("result.exr");

  return 0;
} // end main()

