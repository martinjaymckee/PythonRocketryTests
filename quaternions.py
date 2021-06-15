from numpy import *


# TODO: Reevaluate the structure and implementation of quaternions with the possible efficiency improvements provided by numpy


class Quaternion:
	@classmethod
	def Pure(cls, x, y, z):
		return cls(0, x, y, z)

	def __init__(self, t, x, y, z):
		self.__t = t
		self.__x = x
		self.__y = y
		self.__z = z

	def __str__( self ):
		returnString = '( '
		returnString = returnString + str( self.__t ) + ', [ '
		returnString = returnString + str( self.__x ) + ', '
		returnString = returnString + str( self.__y ) + ', '
		returnString = returnString + str( self.__z ) + ' ] )'
		return returnString

	def __add__( self, other ):
		t = 0.0
		x = 0.0
		y = 0.0
		z = 0.0
		if isinstance( other, float ):
			t = other
		elif isinstance( other, Quaternion ):
			t = self.__t + other.__t
			x = self.__x + other.__x
			y = self.__y + other.__y
			z = self.__z + other.__z
		else:
			print('Type error in Quaternion.__add__')
		return Quaternion( t, x, y, z )

	def __sub__( self, other ):
		t = 0.0
		x = 0.0
		y = 0.0
		z = 0.0
		if isinstance( other, float ):
			t = other
		elif isinstance( other, Quaternion ):
			t = self.__t - other.__t
			x = self.__x - other.__x
			y = self.__y - other.__y
			z = self.__z - other.__z
		else:
			print('Type error in Quaternion.__sub__')
		return Quaternion( t, x, y, z )

	def __mul__( self, other ): # Grassman Product
		t = 0.0
		x = 0.0
		y = 0.0
		z = 0.0
		if isinstance( other, float ) or isinstance(other, int):
			t = self.__t * other
			x = self.__x * other
			y = self.__y * other
			z = self.__z * other
		elif isinstance( other, Quaternion ):
			u0 = self.__t; u1 = self.__x; u2 = self.__y; u3 = self.__z
			v0 = other.__t; v1 = other.__x; v2 = other.__y; v3 = other.__z
			t = ( u0 * v0 ) - ( u1 * v1 + u2 * v2 + u3 * v3 )
			x = u0 * v1 + v0 * u1 + ( u2 * v3 - u3 * v2 )
			y = u0 * v2 + v0 * u2 - ( u1 * v3 - u3 * v1 )
			z = u0 * v3 + v0 * u3 + ( u1 * v2 - u2 * v1 )
		else:
			print('Type error in Quaternion.__mul__, other is a {}'.format(type( other ) ))
		return Quaternion( t, x, y, z )

	def __rmul__( self, other ): # Grassman Product
		t = 0.0
		x = 0.0
		y = 0.0
		z = 0.0
		if isinstance( other, float ) or isinstance(other, int):
			t = self.__t * other
			x = self.__x * other
			y = self.__y * other
			z = self.__z * other
		elif isinstance( other, Quaternion ):
			u0 = self.__t; u1 = self.__x; u2 = self.__y; u3 = self.__z
			v0 = other.__t; v1 = other.__x; v2 = other.__y; v3 = other.__z
			t = ( u0 * v0 ) - ( u1 * v1 + u2 * v2 + u3 * v3 )
			x = u0 * v1 + v0 * u1 + ( u2 * v3 - u3 * v2 )
			y = u0 * v2 + v0 * u2 - ( u1 * v3 - u3 * v1 )
			z = u0 * v3 + v0 * u3 + ( u1 * v2 - u2 * v1 )
		else:
			print('Type error in Quaternion.__mul__, other is a {}'.format(type( other ) ))
		return Quaternion( t, x, y, z )

	def __div__( self, other ):
		t = 0.0
		x = 0.0
		y = 0.0
		z = 0.0
		inverse = None
		if isinstance( other, float ):
			t = other
			inverse = Quaternion( 1.0 / t, x, y, z )
		elif isinstance( other, Quaternion ):
			inverse = other.inverse()
		return self.__mul__( inverse )

	def __truediv__( self, other ):
		return self.__div__( other )

	def conjugate( self ):
		x = self.__x * -1.0
		y = self.__y * -1.0
		z = self.__z * -1.0
		return Quaternion( self.__t, x, y, z )

	def inverse( self ):
		conjugate = self.conjugate()
		norm = pow( ( self.norm() ).s(), -1 )
		return Quaternion(  norm * conjugate.s(), norm * conjugate.x(), norm * conjugate.y(), norm * conjugate.z() )

	def norm( self ):
		sum = pow( self.__t, 2 )
		sum = sum + pow( self.__x, 2 )
		sum = sum + pow( self.__y, 2 )
		sum = sum + pow( self.__z, 2 )
		return Quaternion(sum, 0.0, 0.0, 0.0 )

	def mod( self ):
		norm = self.norm()
		return Quaternion( pow( norm.s(), .5 ), 0.0, 0.0, 0.0 )

	def det( self ):
		norm = self.norm()
		return Quaternion( pow( norm.s(), 2 ), 0.0, 0.0, 0.0 )

	def real( self ):
		return self.__t

	def unreal( self ):
		return self.__x, self.__y, self.__z

	def unit( self ):
		return self * ( self.mod() ).inverse()

	def s( self, s = None ):
		if not s == None:
			self.__t = s
		return self.real()

	def x( self, x = None ):
		if not x == None:
			self.__x = x
		return self.__x

	def y( self, y = None ):
		if not y == None:
			self.__y = y
		return self.__y

	def z( self, z = None ):
		if not z == None:
			self.__z = z
		return self.__z

def quaternionDerivitive(q, omega):
	return 0.5 * Quaternion.Pure(omega[0], omega[1], omega[2]) * q

if __name__ == '__main__':
	valA = Quaternion( 1.0, 0.0, 0.0, 0.0 )
	valB = Quaternion( 0.0, 10.0, 10.0, 10.0 )
	valC = Quaternion( .5, .33, .5, .33 )
	print('Value #1 = ', str( valA ) )
	print('Value #2 = ', str( valB ) )
	print('Value #3 = ', str( valC ) )
	print('Value #1 - Real = ', str( valA.real() ) )
	print('Value #2 - Unreal = ', str( valB.unreal() ) )
	print('Value #3 - Inverse = ', str( valC.inverse() ) )
	print('Value #3 * Value #3 - Inverse = ', str( valC * valC.inverse() ) )
	print('Value #3 - Inverse * Value #3 = ', str( valC.inverse() * valC ) )
	valD = valC.unit()
	print('Value #4 - Unit Quaternion of Value #3 - Value #3 * inv( norm( Value #3 ) ) = ', str( valD ) )
	print('Norm of Value #4 = ', str( valD.norm() ) )
	print('Value #2 * Value #3 = ', str( valB * valC ) )
	print('Value #3 * Value #2 = ', str( valC * valB ) )
	print('Value #1 - Value #2 = ', str( valA - valB ) )
	print('Value #1 + Value #2 = ', str( valA + valB ) )
	print('Value #2 - Value #1 = ', str( valB - valA ) )
	print('Value #2 + Value #1 = ', str( valB + valA ) )
	print(Quaternion.Pure(0, 2, 4).unit())
	print(quaternionDerivitive(valC.unit(), (1, 0.5, 0.25)))
