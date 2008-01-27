/*! \file ImportanceApi.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to an api for the
 *         importance library.
 */

#ifndef IMPORTANCE_API_H
#define IMPORTANCE_API_H

#include "../api/Gotham.h"
class ScalarImportance;

class ImportanceApi
{
  public:
    /*! This method creates a new ScalarImportance object
     *  given the attributes present in the given AttributeMap.
     *  \param attr The set of attributes describing the current
     *              rendering state.
     *  \return A new ScalarImportance object described by the given
     *          attributes.
     */
    static ScalarImportance *importance(Gotham::AttributeMap &attr);

    /*! This method fills an AttributeMap with this library's defaults.
     *  \param attr The set of attributes to add to.
     */
    static void getDefaultAttributes(Gotham::AttributeMap &attr);
}; // end ImportanceApi

#endif // IMPORTANCE_API_H

