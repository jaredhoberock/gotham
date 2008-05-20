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

  // mark all entries taller than the average
  std::vector<size_t> rich;
  for(size_t i = 0;
      i != mTable.size();
      ++i)
  {
    if(mTable[i].mDivide > mean)
    {
      rich.push_back(i);
    } // end if
  } // end for i

  // steal from the rich
  for(size_t i = 0;
      i != mTable.size();
      ++i)
  {
    Real diff = mean - mTable[i].mDivide;
    if(diff > 0)
    {
      if(rich.size() > 0)
      {
        // steal from the first rich guy
        size_t victim = rich.back();

        // set the alias to point to the victim
        mTable[i].mAlias = mTable[victim].mValue;

        // note the alias's pdf
        mTable[i].mAliasPdf = mTable[victim].mValuePdf;

        // steal
        mTable[victim].mDivide -= diff;

        // is he still rich?
        if(mTable[victim].mDivide <= mean)
        {
          rich.pop_back();
        } // end if
      } // end if
      else
      {
        // this case may occur due to floating point error
        // when diff ~ 0
        // fear not, it will still work
      } // end else
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

