#pragma once

#include <map>
#include <memory>
#include <list>

// serialization
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/serialization/weak_ptr.hpp>
#include <boost/serialization/list.hpp>

#include <Core/util.h>

template < typename T >
class GraphNode : public std::enable_shared_from_this< GraphNode< T > >
{
public:
  using ptr = std::shared_ptr< GraphNode< T > >;
  using weak_ptr = std::weak_ptr< GraphNode< T > >;

private:
  GraphNode() = default; // added to use with serialization

  GraphNode( const T & data )
    : id_( 0 )
    , graphId_( graphCounter_ )
    , depth_( 0 )
    , data_( data )
  {
    graphCounter_++;
  }

  GraphNode( const weak_ptr & parent, uint id, const T & data )
    : id_( id )
    , graphId_( parent.lock()->graphId_ )
    , depth_( parent.lock()->depth_ + 1 )
    , data_( data )
  {
    parents_.push_back( parent );
  }

public:
  static ptr root( const T & data ) { graphCounter_++; return ptr( new GraphNode< T >( data ) ); }

  bool isRoot() const { return parent() == nullptr; }
  std::list< ptr > children() const { return children_; }
  ptr parent() const
  {
    if( parents_.size() == 0 )
    {
      return nullptr;
    }
    else
    {
      CHECK_EQ( parents_.size(), 1, "multiple parents, ambiguous call!" );
      return parents_.front().lock();
    }
  }
  std::list< weak_ptr > parents() const { return parents_; }
  std::list< ptr > siblings() const
  {
    for( const auto _parent : parents_ )
    {
      const auto parent = _parent.lock();
      std::list< ptr > siblings;

      for( auto s : parent->children() )
      {
        if( s != this->shared_from_this() )
        {
          siblings.push_back( s );
        }
      }

      return siblings;
    }
  }
  uint id() const { return id_; }
  void setId( uint id ) { id_ = id; }
  uint depth() const { /*CHECK(parents_.size() == 0 || parents_.size() == 1, "ambiguous call to a node depth in a graph");*/ return depth_; }
  T data() const { return data_; }
  T& data() { return data_; }

  ptr makeChild( const T & data, uint id )
  {
    auto child = makeChild(data);
    child->id_ = id;
    return child;
  }

  ptr makeChild( const T & data )
  {
    auto counter = ++counter_[ graphId_ ];

    auto child = ptr( new GraphNode< T >( this->shared_from_this(), counter, data ) );
    children_.push_back( child );

    return child;
  }

  void addExistingChild( const ptr & child )
  {
    children_.push_back( child );
    child->addParent( this->shared_from_this() );
  }

  void removeChild( const ptr & child )
  {
    child->removeParent( this->shared_from_this() );
    children_.remove( child );
  }

  void clearChildren()
  {
    while( ! children_.empty() )
    {
      removeChild( children_.front() );
    }
    //children_.clear();
  }

  friend class boost::serialization::access;
  // When the class Archive corresponds to an output archive, the
  // & operator is defined similar to <<.  Likewise, when the class Archive
  // is a type of input archive the & operator is defined similar to >>.
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version)
  {
    ar & id_;
    ar & depth_;
    ar & graphId_;
    ar & parents_;
    ar & data_;
    ar & children_;
  }

public:
  static std::map< uint, uint > counter_;
  static uint graphCounter_;

private:
  void addParent( const ptr & parent )
  {
    parents_.push_back( parent );
  }

  void removeParent( const weak_ptr & parent )
  {
    parents_.remove_if( [parent](weak_ptr p){ // no standard comparison between weak_ptr
      ptr sparent = parent.lock();
      ptr sp = p.lock();
      if(sparent && sp)
          return sparent == sp;
      return false;
    });
  }

private:
  uint id_;
  uint depth_;
  uint graphId_;
  std::list< weak_ptr > parents_;
  T data_;
  std::list< ptr > children_;
};

template < typename T > std::map< uint, uint > GraphNode< T >::counter_;
template < typename T > uint GraphNode< T >::graphCounter_ = 0;
