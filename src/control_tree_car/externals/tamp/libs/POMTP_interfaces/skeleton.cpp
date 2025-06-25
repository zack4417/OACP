/*  ------------------------------------------------------------------
    Copyright 2016 Camille Phiquepal
    email: camille.phiquepal@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or (at
    your option) any later version. This program is distributed without
    any warranty. See the GNU General Public License for more details.
    You should have received a COPYING file of the full GNU General Public
    License along with this program. If not, see
    <http://www.gnu.org/licenses/>
    --------------------------------------------------------------  */

#include "skeleton.h"

#include <queue>
#include <iostream>
#include <set>
#include <skeleton_printer.h>

static int PolicyNumber = 0;

//----Free functions-------------------//
static std::list< Policy::GraphNodeTypePtr> nodes( const Policy & a )
{
  std::list< Policy::GraphNodeTypePtr > nlist;

  std::queue< Policy::GraphNodeTypePtr > Q;

  if( ! a.empty() )
  {
    Q.push( a.root() );
    nlist.push_back( a.root() );
  }

  while( ! Q.empty() )
  {
    auto n = Q.front();
    Q.pop();

    for( const auto & c : n->children() )
    {
      Q.push( c );
      nlist.push_back( c );
    }
  }

  return nlist;
}

//----QResult-------------------------//
bool operator==(const QResult & a, const QResult & b)
{
  return ( a.stepsPerPhase_ == b.stepsPerPhase_ ) && ( a.world_to_q_list_ == b.world_to_q_list_ ) && ( a.qmask_ == b.qmask_ );
}

//----Policy-------------------------//
Policy::Policy()
  : status_( SKELETON )
  , id_( PolicyNumber )
{

}

Policy::Policy( const GraphNodeTypePtr & root )
  : status_( SKELETON )
  , id_( PolicyNumber )
  , root_( root )
{
  PolicyNumber++;

  // reconstruct the leafs from root
  std::list < GraphNodeTypePtr > Q;
  Q.push_back( root );

  while ( ! Q.empty() )
  {
    auto n = Q.front();
    Q.pop_front();

    if( n->children().size() == 0 )
    {
      leafs_.push_back( n );
    }
    else
    {
      for( auto c : n->children() )
      {
        Q.push_back( c );
      }
    }
  }
}

Policy::Policy( const Policy & policy )
{
  copy( policy );
}

Policy & Policy::operator= ( const Policy & policy )
{
  copy( policy );

  return *this;
}

uint Policy::nNodes() const
{
  std::set< uint > node_ids;

  for(const auto l : leafs_)
  {
    for(const auto n : getPathTo(l.lock()))
    {
      node_ids.insert(n->id());
    }
  }

  return node_ids.size();
}

std::list<Policy::GraphNodeTypePtr> Policy::leaves() const
{
  std::list<Policy::GraphNodeTypePtr> leaves;
  for( const auto& l : leafs_ )
  {
    leaves.push_back(l.lock());
  }

  return leaves;
}

std::list<Policy::GraphNodeTypePtr> Policy::sleaves() const
{
  auto leaves = this->leaves();

  leaves.sort([](Policy::GraphNodeTypePtr a, Policy::GraphNodeTypePtr b)->bool
  {return a->id() < b->id();});

  return leaves;
}

void Policy::save( const std::string & file ) const
{
  std::ofstream ofs( file );
  boost::archive::text_oarchive oa(ofs);
  oa << *this;
}

void Policy::load( const std::string & file )
{
  std::ifstream ifs( file );
  boost::archive::text_iarchive ia(ifs);
  ia >> *this;
}

size_t Policy::hash() const
{
  std::list< Policy::GraphNodeTypePtr > nodesA = nodes( *this );

  std::size_t hash = 0;

  for( const auto n : nodesA )
  {
    hash += 2 << n->depth() + n->data().decisionGraphNodeId;
  }

  return hash;
}

void Policy::saveToGraphFile( const std::string & filename ) const
{
  if( ! root_ )
  {
    return;
  }

  std::ofstream file;
  file.open( filename );

  PolicyPrinter printer( file );
  printer.print( *this );

  file.close();

  // png
  std::string nameCopy( filename );
  const std::string ext( ".gv" );
  std::string newName = nameCopy.replace( nameCopy.find( ext ), ext.length(), ".png" );

  std::stringstream ss;
  ss << "dot"   << " ";
  ss << "-Tpng" << " ";
  ss << "-o"    << " ";
  ss << newName << " ";
  ss << filename;

  system( ss.str().c_str() );
}

void Policy::saveAll( const std::string & folder, const std::string & suffix ) const
{
  std::stringstream namess, skenamess;
  namess << folder << "/" << "policy-" << id() << suffix;
  auto name = namess.str();

  save( name + ".po" );
  saveToGraphFile( name + ".gv" );
}

void Policy::copy( const Policy & policy )
{
  if( policy.root_ )
  {
    id_ = policy.id_;
    value_ = policy.value_;
    qr_ = policy.qr_;
    status_ = policy.status_;
    leafs_.clear();

    auto rootData = policy.root_->data();

    root_ = GraphNodeType::root( rootData );

    std::queue< std::pair < GraphNodeTypePtr, GraphNodeTypePtr > > Q;

    Q.push( std::make_pair( policy.root_, root_ ) ); // original - copy

    while( ! Q.empty() )
    {
      auto u = Q.front();
      Q.pop();

      auto uOriginal = u.first;
      auto uCopy     = u.second;

      for( auto v : uOriginal->children() )
      {
        auto vCopy = uCopy->makeChild( v->data(), v->id() );

        Q.push( std::make_pair( v, vCopy ) );

        if( v->children().size() == 0 )
        {
          leafs_.push_back( vCopy );
        }
      }
    }
  }
}

bool operator== ( const Policy & a, const Policy & b )
{
//  return a.hash() == b.hash();
  std::list< Policy::GraphNodeTypePtr > nodesA = nodes( a );
  std::list< Policy::GraphNodeTypePtr > nodesB = nodes( b );

  bool equal = true;
  if( nodesA.size() != nodesB.size() )
  {
    equal = false;
  }
  else if( nodesA.empty() )
  {
    return true;
  }
  else
  {
    auto itA = nodesA.begin();
    auto itB = nodesB.begin();

    while( itA != nodesA.end() && itB != nodesB.end() )
    {
      equal = equal && (*itA)->data().decisionGraphNodeId == (*itB)->data().decisionGraphNodeId;

      ++itA;
      ++itB;
    }
  }

  return equal;
}
bool operator!= ( const Policy & a, const Policy & b )
{
  return ! ( a == b );
}

std::list< Policy::GraphNodeTypePtr > getPathTo( const Policy::GraphNodeTypePtr & node )
{
  std::list< Policy::GraphNodeTypePtr > path;

  auto n = node;

  path.push_front( n );

  while( n->parent() )
  {
    path.push_front( n->parent() );
    n = n->parent();
  }

  return path;
}
