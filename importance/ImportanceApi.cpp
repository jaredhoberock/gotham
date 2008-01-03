/*! \file ImportanceApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ImportanceApi class.
 */

#include "ImportanceApi.h"
#include "ConstantImportance.h"
#include "LuminanceImportance.h"
#include "NormalizedImportance.h"
#include "InverseLuminanceImportance.h"
#include "EqualVisitImportance.h"
#include "MultipleImportance.h"
#include "LuminanceOverVisits.h"
#include "ExponentImportance.h"
#include "ThroughputLuminanceImportance.h"
#include "ManualImportance.h"
#include "TargetImportance.h"
#include "MaxImportance.h"
using namespace boost;

ScalarImportance *ImportanceApi
  ::importance(const Gotham::AttributeMap &attr)
{
  ScalarImportance *result = 0;
  std::string importanceName = "luminance";

  // fish out the parameters
  Gotham::AttributeMap::const_iterator a = attr.find("importance::function");
  if(a != attr.end())
  {
    any val = a->second;
    importanceName = any_cast<std::string>(val);
  } // end if

  float k = 1.0f;
  a = attr.find("importance::exponent");
  if(a != attr.end())
  {
    any val = a->second;
    k = static_cast<float>(atof(any_cast<std::string>(val).c_str()));
  } // end if

  std::string filterName("bilinear");
  a = attr.find("importance::visitfilter");
  if(a != attr.end())
  {
    any val = a->second;
    filterName = any_cast<std::string>(val);
  } // end if

  bool doFilter = true;
  if(filterName == "bilinear")
  {
    doFilter = true;
  } // end if
  else if(filterName == "nearestneighbor")
  {
    doFilter = false;
  } // end if
  else
  {
    std::cerr << "Warning: unknown visit filter \"" << filterName << "\"." << std::endl;
    doFilter = true;
  } // end else

  // create the importance
  if(importanceName == "luminance")
  {
    result = new LuminanceImportance();
  } // end if
  else if(importanceName == "multiple")
  {
    result = new MultipleImportance();
  } // end else if
  else if(importanceName == "normalized")
  {
    result = new NormalizedImportance();
  } // end else if
  else if(importanceName == "constant")
  {
    result = new ConstantImportance();
  } // end else if
  else if(importanceName == "inverseluminance")
  {
    result = new InverseLuminanceImportance();
  } // end else if
  else if(importanceName == "equalvisit")
  {
    result = new EqualVisitImportance(doFilter);
  } // end else if
  else if(importanceName == "luminanceovervisits")
  {
    result = new LuminanceOverVisits(doFilter);
  } // end else if
  else if(importanceName == "exponent")
  {
    result = new ExponentImportance(k);
  } // end else if
  else if(importanceName == "throughputluminance")
  {
    result = new ThroughputLuminanceImportance();
  } // end else if
  else if(importanceName == "manual")
  {
    result = new ManualImportance();
  } // end else if
  else if(importanceName == "max")
  {
    result = new MaxImportance();
  } // end else if
  else
  {
    std::cerr << "Warning: unknown importance function \"" << importanceName << "\"." << std::endl;
    result = new LuminanceImportance();
  } // end else

  return result;
} // end ImportanceApi::importance()

