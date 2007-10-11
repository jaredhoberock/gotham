/*! \file ScalarImportance.inl
 *  \author Jared Hoberock
 *  \brief Inline file for ScalarImportance.h.
 */

#include "ScalarImportance.h"

float ScalarImportance
  ::getNormalizationConstant(void) const
{
  return mNormalizationConstant;
} // end ScalarImportance::getNormalizationConstant()

float ScalarImportance
  ::getInvNormalizationConstant(void) const
{
  return mInvNormalizationConstant;
} // end ScalarImportance::getInvNormalizationConstant()

