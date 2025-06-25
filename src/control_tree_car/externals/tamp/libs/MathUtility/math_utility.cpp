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

#include <math_utility.h>

double norm2( const arr & x )
{
  return sqrt( ( ( ~ x ) * x )( 0 ) );
}

arr Jnorm2( const arr & x )
{
  arr J( 1, x.N );

  // compute sqrNorm
  double norm = norm2( x );

  // compute each jacobian element
  if( norm > 0.000001 )
  {
    for( auto i = 0; i < x.N; ++i )
      J( 0, i ) = x( i ) / norm ;
  }
  else
  {
    J.setZero();
  }

  return J;
}

double dot( const arr & a, const arr & b )
{
  CHECK( a.d0 == b.d0 && a.d1 == 0 && b.d1 == 0, "wrong dimensions, scalar product is implemented here for vectors only" );

  double scalar_product = 0;
  for( auto i = 0; i < a.N; ++i )
  {
    scalar_product += a( i ) * b( i );
  }

  return scalar_product;
}

arr normalizedX( const arr & x, const arr & Jx,  arr & Jx1 )
{
  double normX = norm2( x );
  arr x1 = x / normX;
  arr JnormX = Jnorm2( x );
  Jx1 = ( Jx * normX - x * JnormX * Jx ) / ( normX * normX ); // jacobian of u normalized

  return x1;
}

//static arr Jdot( const arr & x )
//{
//  // Jacobian of the scalar with x
//  arr J( 1, x.N );

//}
