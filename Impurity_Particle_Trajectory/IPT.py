"""Impurity Particle Trajectory"""

import numpy as np
import ase.data
import scipy
atomic_masses = { k: ase.data.atomic_masses[v] for k,v in ase.data.atomic_numbers.items() }

mp=1.672621e-27 # mass proton, [mp]=kg
mn=1.674927e-27 # mass neutron, [mn]=kg
me=9.109382e-31 # mass electron, [me]=kg
m_amu=1.660539e-27 # Dalton, [m_amu]=kg
qe=1.602176e-19 # elementary charge, [qe]=C
kB=1.380e-23
from scipy.interpolate import RegularGridInterpolator

class Particle:
    """
    symbol is chemical symbol of the Particle element - asume only atoms (currently not molecules)
    positions: input as [x,y,z], SI
    velocity: [vx,vy,vz], SI
    later: also implement where position in cylindrical space [R,Z,phi], vel_cyl = [vR,vZ,vphi]
    """
    def __init__(self,symbol:str,position:np.array,velocity:np.array,t=0):
        assert isinstance(symbol,str) and any(x==symbol for x in atomic_masses), "must be symbol in the chemical symbol in ase.data.atomic_masses database"
        assert (isinstance(position,np.ndarray) or isinstance(position,list)) and len(position)==3, "position must be numpy array of len 3 "
        assert (isinstance(velocity,np.ndarray) or isinstance(velocity,list)) and len(velocity) ==3, "velocity must be numpy array of len 3 "

        self.symbol=symbol
        self.position = np.array(position,dtype=np.float64) # does nothing if allready np.ndarray
        self.velocity= np.array(velocity,dtype=np.float64) # does nothing if allready np.ndarray
        self.mass=atomic_masses[symbol]*m_amu # SI, kg
        self.charge=1*qe # assume we start with 1 ionization
        self.t=t

    def get_position(self):
        return self.position
    
    def get_velocity(self):
        return self.velocity
    
    def get_y(self):
        return np.hstack((self.position,self.velocity))

    def __repr__(self):
        template="Particle(symbol='{}', position=({:.3f}, {:.3f}, {:.3f}), velocity=({:.3f}, {:.3f}, {:.3f}) )"
        return template.format(self.symbol, 
                               self.position[0],
                               self.position[1],
                               self.position[2],
                               self.velocity[0],
                               self.velocity[1],
                               self.velocity[2])


class ParticleSystem(list):  ## We inherit all the methods from built-in class: list
    # no __init__  # because we do not want to add onto the list init method # future: possibly add some init check if the elements in list=Particle object
    # def __init__():
    #     assert isinstance(x==Particle for x in self)
    
    def get_positions(self):
        "Return the positions of all particles as an Nx3 array"
        return np.array([a.position for a in self])
    def set_positions(self, pos):
        "Set the postions of all particles"
        assert len(pos) == len(self) and isinstance(pos,np.ndarray), "input have same length N as number of Particles in ParticleSystem and be np.ndarray"
        for i, p in enumerate(pos):
            self[i].position = p
        # for a, p in zip(self, pos):
        #     a.position = p
    def get_symbols(self):
        "Return a list of the chemical symbols of all atoms"
        return [a.symbol for a in self]
    def get_velocities(self):
        "Return the velocities of all atoms as an Nx3 array"
        return np.array([a.velocity for a in self])
    def set_velocities(self, vel):
        "Set the postions of all atoms"
        assert len(vel) == len(self)
        for i, v in enumerate(vel):
            self[i].velocity = v
    def get_masses(self):
        "Get the mass of all atoms"
        return np.array([a.mass for a in self])
    def get_charges(self):
        "Get the charge of all atoms"
        return np.array([a.charge for a in self])
    def __repr__(self):
        "The representation - for brevity include only symbols"
        template = "Particles(N={}, symbols: {})"
        all_symbols = " ".join(self.get_symbols())
        return template.format(len(self), all_symbols)
    # later:
    # def set_charges(self,pm:bool)
    # restrict to only +- 1 qe
    # assert len(pm)== len(self)
    # for i,pm in enumerate(pm)
    #   if pm==1: # possibly change to ... some interval dep on probability of ionization
        #   self[i].charge += 1
    #   elif:
    #       self[i].charge += -1
    #   else: 
    #        pass
    # 
    # def write_to_file(self, filename, append=False): # not really needed 
    #     if append:
    #         mode = "at"
    #     else:
    #         mode = "wt"
    #     f = open(filename, mode)
    #     print(len(self), file=f)
    #     print("Particle", file=f)
    #     template = "{}  {:.3f} {:.3f} {:.3f}"
    #     for a in self:
    #         print(template.format(a.symbol, a.position[0], a.position[1], a.position[2]), file=f)
    #     f.close()



class f:
    """Acceleration due to the lorentz force of given B(t,r), v, and E(t,r) 
    possibly later a time dependent charge could be implemented as well q(t).
    Assumes y, i.e. position r and velocity v given in Cartesian system that has its z axis aligned with cylinderical z axis
    Solves the equation of motion in this cartesian basis.
    """
    def __init__(self,B_choice:int,B_R0,R0=None,FELTOR_data=None):
        """
        B_choice indicates which B function to use in calculation:
            B_choice == 0: constant B field
            B_choice == 1: analytical 1/R B field
            B_choice == 2: cartesian gradB version: B = e_y *B_R0* 1/x dependency
            B_choice == 3: B interpolated on a data specified in RZvarphi_data

        if B_choice==3: still just chose some random value for B_R0 - not the prettiest...

        FELTOR_data: Dataset of FELTOR data, tuple of numpy vectors, format = (R_grid,Z_grid,varphi_grid,B_R,B_Z,B_varphi)
        - in the future: implement also the E field
        
        R0 is radial coordinate of magnetic axis w. respect to symmetry axis
        R0 is per default set to None such that it can be ignored (it is not used in constant B field function)
        B_R0 is the B field strength at magnetic axis in SI, [B_R0]=T 
        """

        assert isinstance(B_choice,int)
        if B_choice==3:
            assert FELTOR_data is not None, "Specify RZvarphi_data"
            R_g=FELTOR_data[0]
            Z_g=FELTOR_data[1]
            varphi_g=FELTOR_data[2]
            BR_g=FELTOR_data[3]
            BZ_g=FELTOR_data[4]
            Bvarphi_g=FELTOR_data[5]

            # BR,BZ,Bvarphi are in the varphi,Z,R format
            self.int_BR=RegularGridInterpolator((varphi_g,Z_g,R_g),BR_g)
            self.int_BZ=RegularGridInterpolator((varphi_g,Z_g,R_g),BZ_g)
            self.int_Bvarphi=RegularGridInterpolator((varphi_g,Z_g,R_g),Bvarphi_g) # contravariant => co-variant by multiplying with R_g

        self.B_choice = B_choice
        self.B_R0=B_R0
        self.R0=R0


    def calculate(self,particle:Particle,dt,dy):
        """ Calculate the lorentz acceleration at given change in position and velocity, dy = np.array([dr,dv])
        Particle object has the current position and velocity
        output is y'= np.array([velocity,acceleration]) 

        I have chosen not to include tj & yj in each step since that is inside the particle object, hence the f(particle, dt,dy)
        dt,dy is then e.g. k/2 and 1/2 k1 in Introduction to numerical methods by M. Holmes, table 1.3

        Will assume dy is given in cartesian
        """
        
        assert isinstance(particle,Particle), "1st argument must be of type Particle"
        assert isinstance(dy,list) or isinstance(dy,np.ndarray)
        dy=np.array(dy)
        dy=dy.reshape(1,6)

        q=particle.charge
        m=particle.mass
        t=particle.t #current time

        r=particle.position.reshape(1,3) # current position
        v=particle.velocity.reshape(1,3) # current velocity # optimize this later.

        dr=dy[0,0:3].reshape(1,3)
        dv=dy[0,3:].reshape(1,3)

        f=np.hstack((v+dv,self.Lorentz(q,m,t+dt,r+dr,v+dv)))

        return f
    
    def Lorentz(self,q,m,t,r,v):
        acceleration=q/m*(self.E(t,r)+np.cross(v,self.B(t,r)))
        return acceleration

    def E(self,t,r):
        " E field, currently just a constant, no r or t dep."
        return np.array([0,0,0]) # SI units, V/m

    def B(self,t,r):
        """The B field which will be chosen is encoded in B_choice
        B field is parameterized by the B field strength at HFS, i.e. R_HFS
        currently all B fields are given in Cartesian coordinates, and torus axis is assumed at (x,y,z)=0
        r is assumed to be in Cartesian basis
        """
        if self.B_choice==0:
            return self.B_const()
        
        if self.B_choice==1:
            return self.B_R_analytic(r)
        
        if self.B_choice==2:
            return self.B_cart_grad(r)
        
        if self.B_choice==3:
            r=r.reshape(-1)
            r_cyl=self.r_cyl(r).reshape(1,3)
            return self.B_interpolate(r_cyl)

    def B_const(self):
        """B field, currently just a constant, no spatial or temporal dependence
        B field strength is given by the B_R0
        """
        return self.B_R0*np.array([0,1,0])
    
    def B_R_analytic(self,r):
        """B field that is R^(-1) dependent - with symmetry axis at (x,y,z)=(0,0,0)
        direction is along e_phi in cylindrical coordinates, i.e. positive direction seen from above tokamak
        where R is the radius from symmetri axis
        Stationary field - no time dependence
        """
        R=(np.sum(r[0,0:2]**2))**0.5 # ret det her
        B_size=self.R0*self.B_R0/(R) # scalar
        return B_size*self.e_phi(r)
        """B_chosen = 1 : B field that is R^(-1) dependent"""

    def B_cart_grad(self,r):
        "B= B(x) e_y - with some cartesian version of the 1/R dep."
        x=r[0,0]
        B_size=self.R0*self.B_R0/x
        return B_size*np.array([0,1,0])
    
    def B_interpolate(self,r_cyl):
        """Assumes r_cyl is in the (R,Z,varphi) basis 
        In interpolater the format of (varphi,Z,R) is needed, reverse the order
        """
        R_t=r_cyl[0,0]
        Z_t=r_cyl[0,1]
        varphi_t=r_cyl[0,2]

        test_pt=np.array([varphi_t,Z_t,R_t])

        BR_interp=self.int_BR(test_pt)[0]
        BZ_interp=self.int_BZ(test_pt)[0]
        Bvarphi_interp=self.int_Bvarphi(test_pt)[0]
        
        # B_cyl=np.array([BR_interp,BZ_interp,Bvarphi_interp])

        # did not seem to work
        # e_x=np.array([1,0,0])
        # e_y=np.array([0,1,0])

        # Bx_interp=np.dot(B_cyl,e_x)+np.dot(B_cyl,e_x)# no need for z, dot will allways be zero
        # By_interp=np.dot(B_cyl,e_y)+np.dot(B_cyl,e_y)# no need for z, dot will allways be zero

        Bx_interp=np.sin(varphi_t)*BR_interp+np.cos(varphi_t)*Bvarphi_interp
        By_interp=np.cos(varphi_t)*BR_interp-np.sin(varphi_t)*Bvarphi_interp
        Bz_interp=BZ_interp # z axis are aligned, so Z=z

        B_interp_cart=np.array([Bx_interp,By_interp,Bz_interp])  # cartesian basis
        return B_interp_cart
    
    def B_interpolate_cyl(self,r_cyl):
        """Assumes r_cyl is in the (R,Z,varphi) basis 
        In interpolater the format of (varphi,Z,R) is needed, reverse the order
        """
        R_t=r_cyl[0,0]
        Z_t=r_cyl[0,1]
        varphi_t=r_cyl[0,2]

        test_pt=np.array([varphi_t,Z_t,R_t])

        BR_interp=self.int_BR(test_pt)[0]
        BZ_interp=self.int_BZ(test_pt)[0]
        Bvarphi_interp=self.int_Bvarphi(test_pt)[0]
        
        B_cyl=np.array([BR_interp,BZ_interp,Bvarphi_interp])
        return B_cyl
    
    def e_s(self,r):
        "radial unit vector in the (s,phi,z) cylindrical coordinate system"
        s=np.array([r[0,0],r[0,1],0])
        return s/np.linalg.norm(s)

    def e_phi(self,r):
        """
        azimuthal angle unit vector in the (s,phi,z) cylindrical coordinate system
        """ 
        return np.cross(self.e_z(),self.e_s(r))
    
    def e_z(self):
        "z unit vector in the (s,phi,z) cylindrical coordinate system, also in (R,Z,varphi) and Cartesian"
        return np.array([0,0,1])

    def e_R(self,r):
        """radial unit vector in the (R,Z,varphi) cylindrical coordinate system
        exactly the same as e_s, just written for the convenience of having a seperate unit vector
        """
        r=r.reshape(1,3) # just get into standard format
        R=np.array([r[0,0],r[0,1],0]) # only the x,y coordinates
        return R/np.linalg.norm(R)

    def e_varphi(self,r):
        """
        azimuthal angle unit vector in the (R,Z,varphi) cylindrical coordinate system
        e_varphi is found via e_R cross e_Z  to avoid problems with arctan and 0 division
        """ 
        return np.cross(self.e_R(r),self.e_z())

    def e_Z(self):
        "Z unit vector in the (R,Z,varphi) cylindrical coordinate system, also in (s,phi,z) and Cartesian"
        return np.array([0,0,1])
    
    def varphi(self,r):
        "assume r is in Cartesian basis, will output angle taken from y axis in negative out of plane direction"
        r=r.reshape(1,3)
        x=r[0,0]
        y=r[0,1]
        pre_angle=np.arctan(x/y)
        if (x<0) and (y<0):
            angle=pre_angle+np.pi
        elif (x<0) and (y>0):
            angle=pre_angle+2*np.pi
        elif (x>0) and (y<0):
            angle=pre_angle+np.pi
        else:
            angle=pre_angle
        return angle
    
    def r_cyl(self,r):
        R=np.dot(self.e_R(r),r)*1e3 # FELTOR uses mm
        Z=np.dot(self.e_Z(),r)*1e3 # FELTOR uses mm
        varphi=self.varphi(r)
        r_cyl=np.array([R,Z,varphi])
        return r_cyl
    
    # def e_x(self,r_cyl):
    #     "assume r_cyl is (R,Z,varphi) in np.shape(1,3)"
    #     varphi=r_cyl[0,2]
    #     e_x=np.sin(varphi)*np.array([1,0,0])+np.cos(varphi)
    #     ret

    # def e_y(self,r_cyl):
    #     "assume r_cyl is (R,Z,varphi) in np.shape(1,3)"
    #     varphi=r_cyl[0,2]
    #     np.sin()




class RK4:
    """
    Classical Runge-Kutta (RK4)
    
    assume we have the differential equation on the form  dy(t)/dx = f(t,y) 
    so that y=[x,x'] where x=np.array([[x1,x2,x3]]), x' = dx/dt
    with some initial condition y(0)=a
    assume equitemporal time grid

    input:
    system = ParticleSystem object
    model = Lorentz object
    timestep in seconds
    """
    def __init__(self,system:ParticleSystem,model,timestep):
        self.model=model
        self.system=system
        self.timestep=timestep

    def run(self,N_steps:int):
        dt=self.timestep
        " run the RK4 algorithm N_steps number of time-steps"
        for i in range(N_steps):
            for (j,particle_j) in enumerate(self.system):
                yj=particle_j.get_y()
                
                k1=dt*self.model.calculate(particle_j,0,np.zeros(6)) 
                k2=dt*self.model.calculate(particle_j,0.5*dt,0.5*k1)
                k3=dt*self.model.calculate(particle_j,0.5*dt,0.5*k2)
                k4=dt*self.model.calculate(particle_j,dt,k3)

                dy=1/6*(k1+2*k2+2*k3+k4) #RK4, then y_{j+1} = yj + dy

                # update t,y
                particle_j.position+=dy[0,0:3]
                particle_j.velocity+=dy[0,3:]
                particle_j.t+=dt 


        