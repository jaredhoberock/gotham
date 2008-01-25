/*! \file NamedPrimitiveImportance.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a ScalarImportance
 *         function which assigns importance to named Primitives.
 */

#ifndef NAMED_PRIMITIVE_IMPORTANCE_H
#define NAMED_PRIMITIVE_IMPORTANCE_H

#include "ScalarImportance.h"
#include <boost/functional/hash.hpp>

class NamedPrimitiveImportance
  : public ScalarImportance
{
  public:
    /*! \typedef Parent
     *  \brief Shorthand.
     */
    typedef ScalarImportance Parent;

    /*! Constructor accepts the name of a Primitive of interest and
     *  an importance factor.
     *  \param name The name of the Primitive of interest.
     *  \param factor The factor to scale importance by.
     */
    NamedPrimitiveImportance(const std::string &name, const float factor);

    float evaluate(const PathSampler::HyperPoint &x,
                   const Path &xPath,
                   const std::vector<PathSampler::Result> &results);

  protected:
    /*! The name of the Primitive of interest.
     */
    std::string mName;

    /*! The hash of mName.
     */
    size_t mNameHash;

    /*! A hasher to create hashes.
     */
    static boost::hash<std::string> mHasher;

    /*! A factor to scale importance.
     */
    float mFactor;
}; // end NamedPrimitiveImportance

#endif // NAMED_PRIMITIVE_IMPORTANCE_H

