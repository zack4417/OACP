#pragma once

#include <fstream>

#include <skeleton.h>

class PolicyPrinter
{
public:
  PolicyPrinter( std::ostream & ss )
    : ss_( ss )
  {

  }

  void print( const Policy & policy );

private:
  void saveGraphFrom( const Policy::GraphNodeType::ptr & node );

private:
  std::ostream & ss_;
};
