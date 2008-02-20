/*! \file RecordApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of RecordApi class.
 */

#include "RecordApi.h"
#include "RenderFilm.h"
#include "VarianceFilm.h"
#include "PhotonRecord.h"
using namespace boost;

void RecordApi
  ::getDefaultAttributes(Gotham::AttributeMap &attr)
{
  attr["record:type"] = "image";
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

  // what kind of record?
  std::string type = attr["record:type"];

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
  if(type == "image")
  {
    if(doVariance && estimate.get() != 0)
    {
      result = new VarianceFilm(width, height, estimate, outfile, varianceOutfile);
    } // end if
    else
    {
      result = new RenderFilm(width, height, outfile);
    } // end else
  } // end if
  else if(type == "photonmap")
  {
    // figure out how many photons we should allocate
    size_t width = lexical_cast<size_t>(attr["record:width"]);
    size_t height = lexical_cast<size_t>(attr["record:height"]);

    // if nothing is specified in renderer:target:count,
    // default to the number of pixels
    size_t numPhotons = width * height;

    // check for the specific target photon count attributes
    std::string targetFunctionName = attr["renderer:target:function"];
    if(targetFunctionName == "photons")
    {
      Gotham::AttributeMap::const_iterator a = attr.find("renderer:target:count");
      if(a != attr.end())
      {
        numPhotons = lexical_cast<size_t>(a->second);
      } // end if
    } // end if

    result = new PhotonRecord(numPhotons, outfile);
  } // end else if
  else
  {
    std::cerr << "Warning: Unknown record type \"" << type << "\". Creating image record." << std::endl;
    if(doVariance && estimate.get() != 0)
    {
      result = new VarianceFilm(width, height, estimate, outfile, varianceOutfile);
    } // end if
    else
    {
      result = new RenderFilm(width, height, outfile);
    } // end else
  } // end else

  return result;
} // end RecordApi::record()

