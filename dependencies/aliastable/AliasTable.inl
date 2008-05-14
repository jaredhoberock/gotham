/*! \file AliasTable.inl
 *  \author Jared Hoberock
 *  \brief Inline file for AliasTable.h.
 */

#include "AliasTable.h"
#include <algorithm>

template<typename Type, typename Real>
  bool AliasTable<Type,Real>
    ::Entry
      ::operator<(const Entry &rhs) const
{
  return mDivide < rhs.mDivide;
} // end Entry::operator<()

template<typename Type, typename Real>
template<typename ElementIterator,
         typename ProbabilityIterator>
  void AliasTable<Type, Real>
    ::build(ElementIterator begin, ElementIterator end,
            ProbabilityIterator beginP, ProbabilityIterator endP)
{
  Real sum = 0, mean = 0;

  mTable.clear();

  unsigned int n = 0;
  ProbabilityIterator p = beginP;
  for(ElementIterator i = begin;
      i != end;
      ++i, ++p)
  {
    mTable.resize(mTable.size() + 1);
    mTable.back().mValue = *i;
    mTable.back().mAlias = *i;
    mTable.back().mDivide = *p;
    mTable.back().mValuePdf = *p;
    mTable.back().mAliasPdf = *p;
    sum += *p;
    ++n;
  } // end for i

  mean = sum / n;

  // sort table in ascending order
  std::sort(mTable.begin(), mTable.end());

  Real diff = 0;
  for(size_t i = 0;
      i != mTable.size();
      ++i)
  {
    diff = mean - mTable[i].mDivide;
    if(diff > 0)
    {
      for(size_t j = i+1;
          j != mTable.size();
          ++j)
      {
        if(mTable[j].mDivide >= diff)
        {
          // steal from the rich
          mTable[i].mAlias = mTable[j].mValue;
          mTable[i].mAliasPdf = mTable[i].mAliasPdf;
          mTable[j].mDivide -= diff;
          break;
        } // end if
      } // end for j
    } // end if
  } // end for i

  Real oneOverMean = static_cast<Real>(1.0) / mean;
  Real oneOverSum = Real(1) / sum;

  // normalize and create inverse map
  mInverseMap.clear();
  for(size_t i = 0;
      i != mTable.size();
      ++i)
  {
    mTable[i].mDivide *= oneOverMean;
    mTable[i].mValuePdf *= oneOverSum;
    mTable[i].mAliasPdf *= oneOverSum;

    mInverseMap[mTable[i].mValue] = mTable[i].mValuePdf;
  } // end for
} // end AliasTable::build()

template<typename Type, typename Real>
  Type &AliasTable<Type,Real>
    ::operator()(const Real &u)
{
  // FIXME: change this to a fast floor
  Real q = Real(mTable.size()) * u;
  size_t i = static_cast<size_t>(q);
  Real u1 = q - i;
  Entry &e = mTable[i];
  return u1 < e.mDivide ? e.mValue : e.mAlias;
} // end AliasTable::operator()()

template<typename Type, typename Real>
  const Type &AliasTable<Type,Real>
    ::operator()(const Real &u) const
{
  // FIXME: change this to a fast floor
  Real q = Real(mTable.size()) * u;
  size_t i = static_cast<size_t>(q);
  Real u1 = q - i;
  const Entry &e = mTable[i];
  return u1 < e.mDivide ? e.mValue : e.mAlias;
} // end AliasTable::operator()()

template<typename Type, typename Real>
  const Type &AliasTable<Type,Real>
    ::operator()(const Real &u,
                 Real &pdf) const
{
  // FIXME: change this to a fast floor
  Real q = Real(mTable.size()) * u;
  size_t i = static_cast<size_t>(q);
  Real u1 = q - i;
  const Entry &e = mTable[i];

  const Type *result;
  if(u1 < e.mDivide)
  {
    result = &e.mValue;
    pdf = e.mValuePdf;
  } // end if
  else
  {
    result = &e.mAlias;
    pdf = e.mAliasPdf;
  } // end else

  return *result;
} // end AliasTable::operator()()

template<typename Type, typename Real>
  size_t AliasTable<Type,Real>
    ::size(void) const
{
  return mTable.size();
} // end AliasTable::size()

template<typename Type, typename Real>
  Real AliasTable<Type,Real>
    ::evaluatePdf(const Type &t) const
{
  Real result = 0;

  typename std::map<Type,Real>::const_iterator s = mInverseMap.find(t);
  if(s != mInverseMap.end())
    result = s->second;

  return result;
} // end AliasTable::evaluatePdf()

template<typename Type, typename Real>
  bool AliasTable<Type, Real>
    ::empty(void) const
{
  return mTable.empty();
} // end AliasTable::empty()

template<typename Type, typename Real>
  const std::vector<typename AliasTable<Type,Real>::Entry> &AliasTable<Type,Real>
    ::getTable(void) const
{
  return mTable;
} // end AliasTable::getTable()

