import math

class Impedance( complex ):
    def __init__( self,  real,  imag = 0 ):
        complex.__init__( self,  real,  imag )
        
    def __str__(self):
        assert False,  'Inpedance.__str__: Method must be implemented in a subclass'
        
    def __add__( self,  other ):
        pass # This should implement a series connection
        
    def __or__( self,  other ): # Check this overload
        pass # This should implement a parallel connection

    def value(self,  f = 0 ):
        return self.real
        
    def resistance(self):
        return self.value()
    
class Resistance( Impedance ):
    def __init__( self,  resistance ):
        Impedance.__init__( self,  resistance,  0.0 )
        
    def __str__(self):
        return str( self.resistance() ) + ' ohms'

class Capacitance(Impedance):
    def __init__( self,  capacitance ):
        Impedance.__init__( self,  0.0,  capacitance )
        self.__capacitance = capacitance
        
    def __str__(self):
        return str( self.capacitance() ) + ' F'

    def capacitance(self):
        return self.__capacitance

class NoiseSource:
    def __init__(self):
        pass
        
    def type(self):
        assert False,  "NoiseSource.type: Method must be overloaded in subclasses to return either 'voltage' or 'current'."
        
    def scaling(self):
        assert False,  "NoiseSource.scaling: Method must be overloaded in subclasses to return the manner in which the noise scales."
        
    def value( self,  frequency ):
        assert False,  "NoiseSource.value: Method must be overloaded in subclasses."        

    def power( self, cutoff_frequency,  low_frequency = None ):
        start_frequency= 0.0
        end_frequency = cutoff_frequency
        if not low_frequency == None:
            start_frequency = low_frequency
        # This needs to produce noise POWER from the value ( voltage ) method...
        #   It may need to drive an impedance or be sourced from a known impedance
        
class TheoryElectronics:
    def __init__( self ):
        self.__numeric_mode = None # Need to be able to set things for the solution of 
        
    def V(self,  R,  I ): return I*R.resistance()
    def I( self,  R,  V ): return V / R.resistance
    def R( self,  I,  V ): return Resistance( V / I )
    
    def parallelImpedance( self,  items ):
        if not len( items ) == 0:
            if isinstance( items[0],  Resistance ):
                resistances = [ item.resistance() for item in items ]
                return Resistance( self.__add_inverse( resistances ) )
            else:
                capacitances = [ item.capacitance() for item in items ]
                return Capacitance( self.__add_direct( capacitances ) )        
        return None
        
    def seriesImpedance( self,  items ):
        if not len( items ) == 0:
            if isinstance( items[0],  Resistance ):
                resistances = [ item.resistance() for item in items ]
                return Resistance( self.__add_direct( resistances ) )
            else:
                capacitances = [ item.capacitance() for item in items ]
                return Capacitance( self.__add_inverse( capacitances ) )
        return None

    def __add_direct( self,  impedances ):
        sum = 0.0
        for impedance in impedances:
            sum += impedance
        return sum
        
    def __add_inverse( self,  impedances ):
        inverse_sum = 0.0
        for impedance in impedances:
            inverse_sum += ( 1.0 / impedance )
        return ( 1.0 / inverse_sum )
        
def johnsonNoiseRMS( R,  bandwidth,  T = 298.0 ):
    Kb = 1.3806504E-23
    return math.sqrt( 4.0 * Kb * T * bandwidth * R.resistance() )
    
def shotNoiseRMS( I,  bandwidth ):
   q = 1.60E-19 # coulombs - Charge of an electron 
   return math.sqrt( 2.0 * q * I * bandwidth )
   
def SNR( signal,  noise ):
    return 20.0 * math.log( signal / noise ) / math.log( 10.0 )
