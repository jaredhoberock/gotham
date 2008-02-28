/*! \file PartitionedPhotonRecord.cpp
 *  \author Jared Hoberock
 *  \brief Implementation of PartitionedPhotonRecord class.
 */

#include "PartitionedPhotonRecord.h"
#include <fstream>

PartitionedPhotonRecord
  ::PartitionedPhotonRecord(void)
    :Parent()
{
  ;
} // end PartitionedPhotonRecord::PartitionedPhotonRecord()

PartitionedPhotonRecord
  ::PartitionedPhotonRecord(const size_t n,
                            const std::string &filename)
    :Parent(n/2)
{
  mCausticMap.reserve(n/2);
} // end PartitionedPhotonRecord::PartitionedPhotonRecord()

void PartitionedPhotonRecord
  ::postprocess(void)
{
  // bypass PhotonRecord's postprocess
  Parent::Parent0::postprocess();

  if(mFilename != "")
  {
    std::fstream outfile(mFilename.c_str(), std::fstream::out);
    if(outfile.is_open())
    {
      // write the global photon map
      outfile << "g.pushAttributes()" << std::endl;
      outfile << "g.attribute(\"name\", \"global\")" << std::endl;
      outfile << *this << std::endl;
      outfile << "g.attribute(\"name\", \"caustic\")" << std::endl;
      outfile << mCausticMap << std::endl;
      outfile << "g.popAttributes()" << std::endl;

      std::cout << "Wrote photons to: " << getFilename() << std::endl;
    } // end if
    else
    {
      std::cout << "Error: Unable to write photons to: " << getFilename() << std::endl;
    } // end else
  } // end if
} // end PhotonRecord::postprocess()

