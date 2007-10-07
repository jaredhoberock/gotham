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

  // create the importance
  if(importanceName == "luminance")
  {
    result = new LuminanceImportance();
  } // end if
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
    result = new EqualVisitImportance();
  } // end else if
  else
  {
    std::cerr << "Warning: unknown importance function \"" << importanceName << "\"." << std::endl;
    result = new LuminanceImportance();
  } // end else

  return result;
} // end ImportanceApi::importance()

