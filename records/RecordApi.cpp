/*! \file RecordApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RecordApi class.
 */

#include "RecordApi.h"
#include "RenderFilm.h"
#include "VarianceFilm.h"
using namespace boost;

void RecordApi
  ::getDefaultAttributes(Gotham::AttributeMap &attr)
{
  attr["record:outfile"] = "gotham.exr";

  attr["record:estimate:infile"] = "";
  attr["record:variance:outfile"] = "";
  attr["record:acceptance:outfile"] = "";
  attr["record:proposals:outfile"] = "";
  attr["record:target:outfile"] = "";
  attr["record:width"] = "512";
  attr["record:height"] = "512";
} // end RecordApi::getDefaultAttributes()

Record *RecordApi
  ::record(Gotham::AttributeMap &attr)
{
  Record *result = 0;

  // name of the output?
  std::string outfile = attr["record:outfile"];

  // name of the variance output?
  std::string varianceOutfile = attr["record:variance:outfile"];

  // name of the estimate image?
  std::string estimateFilename = attr["record:estimate:infile"];

  // should we produce a variance image?
  bool doVariance = (varianceOutfile != std::string(""));

  // load the estimate
  boost::shared_ptr<RandomAccessFilm> estimate;
  if(estimateFilename != std::string(""))
  {
    estimate.reset(new RandomAccessFilm());

    // XXX check to make sure this read succeeded
    estimate->readEXR(estimateFilename.c_str());
  } // end if

  // image width
  size_t width  = lexical_cast<size_t>(attr["record:width"]);
  size_t height = lexical_cast<size_t>(attr["record:height"]);

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

