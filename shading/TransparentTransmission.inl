/*! \file TransparentTransmission.inl
 *  \author Jared Hoberock
 *  \brief Inline file for TransparentTransmission.h.
 */

#include "TransparentTransmission.h"

TransparentTransmission
  ::TransparentTransmission(const Spectrum &transmittance)
    :Parent(),mTransmittance(transmittance)
{
  ;
} // end TransparentTransmission::TransparentTransmission()

