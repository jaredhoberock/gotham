/*! \file RecordApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an api
 *         for instancing Records.
 */

#ifndef RECORD_API_H
#define RECORD_API_H

#include "Record.h"
#include "../api/Gotham.h"

class RecordApi
{
  public:
    /*! This static method creates a new Record given
     *  the options in the given AttributeMap.
     *  \param attr An AttributeMap describing a set of
     *              recording attributes.
     */
    static Record *record(const Gotham::AttributeMap &attr);
}; // end RecordApi

#endif // RECORD_API_H

