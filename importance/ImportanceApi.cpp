/*! \file ImportanceApi.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of ImportanceApi class.
 */

#include "ImportanceApi.h"
#include "ConstantImportance.h"
#include "LuminanceImportance.h"
#include "NormalizedImportance.h"
#include "InverseLuminanceImportance.h"
#include "ThroughputLuminanceImportance.h"
#include "ManualImportance.h"
#include "TargetImportance.h"
#include "MaxImportance.h"
#include "NamedPrimitiveImportance.h"
#include <boost/lexical_cast.hpp>
using namespace boost;

void ImportanceApi
  ::getDefaultAttributes(Gotham::AttributeMap &attr)
{
  attr["importance:function"] = "luminance";

  attr["importance:namedprimitive:name"] = "";

  attr["importance:namedprimitive:factor"] = "1.0";
} // end ImportanceApi::getDefaultAttributes()

ScalarImportance *ImportanceApi
  ::importance(Gotham::AttributeMap &attr)
{
  ScalarImportance *result = 0;

  // fish out the parameters
  std::string importanceName = attr["importance:function"];

  std::string primitiveName = attr["importance:namedprimitive:name"];

  float namedPrimitiveFactor = lexical_cast<float>(attr["importance:namedprimitive:factor"]);

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
  else if(importanceName == "namedprimitive")
  {
    result = new NamedPrimitiveImportance(primitiveName, namedPrimitiveFactor);
  } // end else if
  else
  {
    std::cerr << "Warning: unknown importance function \"" << importanceName << "\"." << std::endl;
    result = new LuminanceImportance();
  } // end else

  return result;
} // end ImportanceApi::importance()

