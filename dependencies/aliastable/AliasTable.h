/*! \file AliasTable.h
 *  \author Jared Hoberock
 *  \brief Defines the interface to a class implementing
 *         an alias table for constant time access to
 *         discrete distributions.
 */

#ifndef ALIAS_TABLE_H
#define ALIAS_TABLE_H

#include <vector>
#include <map>

template<typename Type, typename Real = float>
  class AliasTable
{
  public:
    template<typename ElementIterator,
             typename ProbabilityIterator>
      void build(ElementIterator begin, ElementIterator end,
                 ProbabilityIterator beginP, ProbabilityIterator endP);
    

    Type &operator()(const Real &u);
    const Type &operator()(const Real &u) const;

    const Type &operator()(const Real &u, Real &pdf) const;

    inline size_t size(void) const;

    Real evaluatePdf(const Type &t) const;

    Real invert(const Type &t) const;

    /*! This method returns whether or not this AliasTable is empty.
     *  \return mTable.empty()
     */
    inline bool empty(void) const;

    struct Entry
    {
      Type mValue, mAlias;
      Real mDivide;
      Real mValuePdf, mAliasPdf;
      inline bool operator<(const Entry &rhs) const;
    }; // end Entry

    /*! This method returns a const reference to mTable.
     *  \return mTable
     */
    inline const std::vector<Entry> &getTable(void) const;

  protected:
    std::vector<Entry> mTable;

    std::map<Type,Real> mInverseMap;
}; // end AliasTable

#include "AliasTable.inl"

#endif // ALIAS_TABLE_H

