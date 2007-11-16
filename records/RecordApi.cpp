/*! \file RecordApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RecordApi class.
 */

#include "RecordApi.h"
#include "RenderFilm.h"
#include "VarianceFilm.h"
using namespace boost;

Record *RecordApi
  ::record(const Gotham::AttributeMap &attr)
{
  Record *result = 0;

  // name of the output?
  std::string outfile = "";

  // XXX deprecate "renderer::outfile"
  Gotham::AttributeMap::const_iterator a = attr.find("renderer::outfile");
  if(a != attr.end())
  {
    any val = a->second;
    outfile = any_cast<std::string>(val).c_str();
    std::cerr << "Warning: attribute \"renderer::outfile\" is deprecated.  Use \"record::outfile\" instead." << std::endl;
  } // end if
  a = attr.find("record::outfile");
  if(a != attr.end())
  {
    any val = a->second;
    outfile = any_cast<std::string>(val).c_str();
  } // end if

  // name of the variance output?
  std::string varianceOutfile = "variance.exr";
  a = attr.find("record::varianceoutfile");
  if(a != attr.end())
  {
    any val = a->second;
    varianceOutfile = any_cast<std::string>(val).c_str();
  } // end if

  // name of the estimate image?
  std::string estimateFilename = "";
  a = attr.find("record::estimatefilename");
  if(a != attr.end())
  {
    any val = a->second;
    estimateFilename = any_cast<std::string>(val).c_str();
  } // end if

  // should we produce a variance image?
  bool doVariance = false;
  a = attr.find("record::estimatevariance");
  if(a != attr.end())
  {
    any val = a->second;
    doVariance = (any_cast<std::string>(val) == std::string("true"));
  } // end if

  // load the estimate
  boost::shared_ptr<RandomAccessFilm> estimate;
  if(estimateFilename != std::string(""))
  {
    estimate.reset(new RandomAccessFilm());

    // XXX check to make sure this read succeeded
    estimate->readEXR(estimateFilename.c_str());
  } // end if

  // image width
  unsigned int width = 512;
  a = attr.find("record::width");
  if(a != attr.end())
  {
    any val = a->second;
    width = atoi(any_cast<std::string>(val).c_str());
  } // end if

  // image height
  unsigned int height = 512;
  a = attr.find("record::height");
  if(a != attr.end())
  {
    any val = a->second;
    height = atoi(any_cast<std::string>(val).c_str());
  } // end if

  // create the Record
  if(doVariance && estimate.get() != 0)
  {
    result = new VarianceFilm(width, height, estimate, outfile, varianceOutfile);
  } // end if
  else
  {
    result = new RenderFilm(width, height, outfile);
  } // end else

  return result;
} // end RecordApi::record()

